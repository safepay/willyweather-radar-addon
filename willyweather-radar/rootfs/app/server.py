#!/usr/bin/env python3
"""WillyWeather Radar Addon for Home Assistant."""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageFilter
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Configuration
CONFIG_PATH = '/data/options.json'
CACHE_DIR = '/data/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Global configuration
config = {}
cache = {}


def load_config():
    """Load configuration from options.json."""
    global config
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        else:
            logger.warning(f"Config file not found at {CONFIG_PATH}, using defaults")
            config = {
                'api_key': os.environ.get('API_KEY', ''),
                'cache_duration': 300,
                'log_level': 'info'
            }
        
        log_level = config.get('log_level', 'info').upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Log level set to: {log_level}")
        
        if not config.get('api_key'):
            logger.error("API key not configured!")
            logger.error("Please configure your WillyWeather API key in the addon configuration")
            sys.exit(1)
        else:
            logger.info(f"API key configured: {config['api_key'][:8]}...")
            
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)


class WillyWeatherAPI:
    """Interface to WillyWeather API."""
    
    BASE_URL = "https://api.willyweather.com.au/v2"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HomeAssistant-WillyWeatherRadar/1.0'
        })
    
    def get_map_providers(self, lat: float, lng: float, map_type: str, 
                         offset: int = -60, limit: int = 60) -> List[Dict]:
        """
        Get map providers for a location.
        
        Args:
            lat: Latitude
            lng: Longitude
            map_type: Type of map (regional-radar or radar, etc.)
            offset: Minutes to start from (negative for past)
            limit: Minutes to end at
            
        Returns:
            List of map provider dictionaries
        """
        url = f"{self.BASE_URL}/{self.api_key}/maps.json"
        
        headers = {
            'Content-Type': 'application/json',
            'x-payload': json.dumps({
                'lat': lat,
                'lng': lng,
                'mapTypes': [{'code': map_type}],
                'offset': offset,
                'limit': limit,
                'verbose': True,
                'units': {
                    'distance': 'km',
                    'speed': 'km/h'
                }
            })
        }
        
        try:
            logger.debug(f"Requesting map providers: {map_type} at ({lat}, {lng})")
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Received {len(data)} providers")
            return data
        except requests.RequestException as e:
            logger.error(f"Failed to get map providers: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            return []
    
    def download_overlay(self, overlay_path: str, overlay_name: str) -> Optional[bytes]:
        """
        Download a radar overlay image.
        
        Args:
            overlay_path: Base path for overlays (already includes protocol)
            overlay_name: Name of the overlay file
            
        Returns:
            Image bytes or None if failed
        """
        # overlay_path already includes 'https:' or '//', don't add protocol
        if overlay_path.startswith('//'):
            url = f"https:{overlay_path}{overlay_name}"
        elif overlay_path.startswith('https:'):
            url = f"{overlay_path}{overlay_name}"
        else:
            url = overlay_path + overlay_name
        
        try:
            logger.debug(f"Downloading overlay: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            logger.debug(f"Downloaded {len(response.content)} bytes")
            return response.content
        except requests.RequestException as e:
            logger.error(f"Failed to download overlay {url}: {e}")
            return None


class RadarBlender:
    """Blend multiple radar images together."""
    
    @staticmethod
    def calculate_coverage(bounds: Dict, center_lat: float, center_lng: float, 
                          zoom_radius_km: float) -> float:
        """
        Calculate how much of the view area is covered by a radar.
        
        Args:
            bounds: Radar bounds dict with minLat, maxLat, minLng, maxLng
            center_lat: View center latitude
            center_lng: View center longitude
            zoom_radius_km: Approximate radius of view in km
            
        Returns:
            Coverage score (0-1, higher is better)
        """
        # Calculate overlap between view box and radar box
        view_min_lat = center_lat - (zoom_radius_km / 111.0)  # Rough conversion
        view_max_lat = center_lat + (zoom_radius_km / 111.0)
        view_min_lng = center_lng - (zoom_radius_km / (111.0 * np.cos(np.radians(center_lat))))
        view_max_lng = center_lng + (zoom_radius_km / (111.0 * np.cos(np.radians(center_lat))))
        
        # Calculate intersection
        inter_min_lat = max(view_min_lat, bounds['minLat'])
        inter_max_lat = min(view_max_lat, bounds['maxLat'])
        inter_min_lng = max(view_min_lng, bounds['minLng'])
        inter_max_lng = min(view_max_lng, bounds['maxLng'])
        
        if inter_min_lat >= inter_max_lat or inter_min_lng >= inter_max_lng:
            return 0.0
        
        # Calculate areas
        inter_area = (inter_max_lat - inter_min_lat) * (inter_max_lng - inter_min_lng)
        view_area = (view_max_lat - view_min_lat) * (view_max_lng - view_min_lng)
        
        return min(inter_area / view_area, 1.0) if view_area > 0 else 0.0
    
    @staticmethod
    def blend_images(images: List[Tuple[bytes, float]], smooth: bool = True) -> Optional[bytes]:
        """
        Blend multiple radar images with weights and optional smoothing.

        Args:
            images: List of (image_bytes, weight) tuples
            smooth: Apply smoothing filters to reduce pixelation (default True)

        Returns:
            Blended image bytes or None
        """
        if not images:
            return None

        try:
            # Load all images
            pil_images = []
            weights = []

            for img_bytes, weight in images:
                img = Image.open(BytesIO(img_bytes)).convert('RGBA')
                pil_images.append(img)
                weights.append(weight)

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)

            # If only one image, apply smoothing and return
            if len(pil_images) == 1:
                result_img = pil_images[0]
                if smooth:
                    # Apply gentle smoothing for single images
                    result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.5))
                output = BytesIO()
                result_img.save(output, format='PNG', optimize=True)
                return output.getvalue()

            # Ensure all images are the same size using high-quality resampling
            base_size = pil_images[0].size
            for i in range(1, len(pil_images)):
                if pil_images[i].size != base_size:
                    # Use BICUBIC for smoother results (better than LANCZOS for radar images)
                    pil_images[i] = pil_images[i].resize(base_size, Image.BICUBIC)

            # Convert to numpy arrays for blending
            arrays = [np.array(img, dtype=np.float32) for img in pil_images]

            # Weighted blending
            blended = np.zeros_like(arrays[0])
            for arr, weight in zip(arrays, weights):
                blended += arr * weight

            # Convert back to image
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            result_img = Image.fromarray(blended, mode='RGBA')

            # Apply smoothing filters to reduce pixelation
            if smooth:
                # Apply gentle Gaussian blur to smooth transitions
                result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.8))

                # Apply slight sharpening to maintain detail after blur
                result_img = result_img.filter(ImageFilter.UnsharpMask(
                    radius=1.0,
                    percent=50,
                    threshold=2
                ))

            # Save to bytes with optimization
            output = BytesIO()
            result_img.save(output, format='PNG', optimize=True)
            return output.getvalue()

        except Exception as e:
            logger.error(f"Failed to blend images: {e}", exc_info=True)
            return None


# Initialize API
api = None


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '1.0.5',
        'api_configured': bool(config.get('api_key'))
    })


@app.route('/api/radar')
def get_radar():
    """
    Get radar image for a location and zoom level.

    Query parameters:
        lat: Latitude
        lng: Longitude
        zoom: Zoom level (higher = more zoomed in)
        timestamp: Optional timestamp for specific overlay

    Returns:
        PNG image with geographic bounds in response headers for Google Maps overlay:
        - X-Radar-Bounds-South: Southern latitude boundary
        - X-Radar-Bounds-West: Western longitude boundary
        - X-Radar-Bounds-North: Northern latitude boundary
        - X-Radar-Bounds-East: Eastern longitude boundary
    """
    try:
        lat = float(request.args.get('lat', -33.8688))
        lng = float(request.args.get('lng', 151.2093))
        zoom = int(request.args.get('zoom', 10))
        timestamp = request.args.get('timestamp')

        # Validate coordinates
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Invalid latitude'}), 400
        if not (-180 <= lng <= 180):
            return jsonify({'error': 'Invalid longitude'}), 400
        if not (1 <= zoom <= 20):
            return jsonify({'error': 'Invalid zoom level'}), 400

        # Determine zoom radius and map type
        # Zoom levels roughly: 5=5000km, 7=1250km, 9=312km, 10=156km, 11=78km, 13=39km
        zoom_radius_km = 5000 / (2 ** (zoom - 5))

        # Use national radar if zoom level <= 10 (>160km radius)
        # Switch to regional radar at zoom 11+ for detailed blending
        use_national = zoom_radius_km > 160
        map_type = 'radar' if use_national else 'regional-radar'

        # Calculate geographic bounds for Google Maps overlay
        # Account for Earth's curvature at different latitudes
        lat_offset = zoom_radius_km / 111.0  # 1 degree latitude ≈ 111km
        lng_offset = zoom_radius_km / (111.0 * np.cos(np.radians(lat)))  # Adjust for latitude

        bounds = {
            'south': lat - lat_offset,
            'west': lng - lng_offset,
            'north': lat + lat_offset,
            'east': lng + lng_offset
        }

        logger.info(f"Radar request: lat={lat}, lng={lng}, zoom={zoom}, "
                   f"radius={zoom_radius_km:.1f}km, type={map_type}")
        
        # Check cache
        cache_key = f"{map_type}_{lat}_{lng}_{zoom}_{timestamp or 'latest'}"
        cache_entry = cache.get(cache_key)
        cache_duration = config.get('cache_duration', 300)
        
        if cache_entry and (time.time() - cache_entry['time']) < cache_duration:
            logger.debug(f"Cache hit for {cache_key}")
            response = send_file(
                BytesIO(cache_entry['data']),
                mimetype='image/png',
                as_attachment=False,
                download_name='radar.png'
            )
            # Add geographic bounds headers for Google Maps compatibility
            response.headers['X-Radar-Bounds-South'] = str(bounds['south'])
            response.headers['X-Radar-Bounds-West'] = str(bounds['west'])
            response.headers['X-Radar-Bounds-North'] = str(bounds['north'])
            response.headers['X-Radar-Bounds-East'] = str(bounds['east'])
            return response
        
        # Get map providers
        providers = api.get_map_providers(lat, lng, map_type)
        
        if not providers:
            logger.warning(f"No radar providers found for {lat}, {lng}")
            return jsonify({'error': 'No radar providers found'}), 404
        
        # For national radar, just use the single provider
        if use_national:
            provider = providers[0]
            overlays = provider.get('overlays', [])
            
            if not overlays:
                logger.warning(f"No overlays available for provider {provider.get('name')}")
                return jsonify({'error': 'No overlays available'}), 404
            
            # Get specific timestamp or latest
            if timestamp:
                overlay = next((o for o in overlays if o['dateTime'] == timestamp), overlays[-1])
            else:
                overlay = overlays[-1]
            
            logger.info(f"Using national radar: {provider.get('name')}, overlay: {overlay['dateTime']}")
            image_data = api.download_overlay(provider['overlayPath'], overlay['name'])

            if not image_data:
                return jsonify({'error': 'Failed to download overlay'}), 500

            # Cache result
            cache[cache_key] = {
                'time': time.time(),
                'data': image_data
            }

            response = send_file(
                BytesIO(image_data),
                mimetype='image/png',
                as_attachment=False,
                download_name='radar.png'
            )
            # Add geographic bounds headers for Google Maps compatibility
            response.headers['X-Radar-Bounds-South'] = str(bounds['south'])
            response.headers['X-Radar-Bounds-West'] = str(bounds['west'])
            response.headers['X-Radar-Bounds-North'] = str(bounds['north'])
            response.headers['X-Radar-Bounds-East'] = str(bounds['east'])
            return response
        
        # For regional radar, blend multiple radars based on coverage
        weighted_images = []
        
        for provider in providers[:5]:  # Limit to 5 closest radars
            # Calculate coverage
            coverage = RadarBlender.calculate_coverage(
                provider['bounds'], lat, lng, zoom_radius_km
            )
            
            if coverage < 0.05:  # Skip if coverage is too low
                logger.debug(f"Skipping {provider['name']}: coverage too low ({coverage:.2%})")
                continue
            
            overlays = provider.get('overlays', [])
            if not overlays:
                logger.debug(f"Skipping {provider['name']}: no overlays")
                continue
            
            # Get specific timestamp or latest
            if timestamp:
                overlay = next((o for o in overlays if o['dateTime'] == timestamp), overlays[-1])
            else:
                overlay = overlays[-1]
            
            image_data = api.download_overlay(provider['overlayPath'], overlay['name'])
            
            if image_data:
                weighted_images.append((image_data, coverage))
                logger.info(f"Added radar {provider['name']} with coverage {coverage:.2%}")
        
        if not weighted_images:
            logger.warning("No radar images available after filtering")
            return jsonify({'error': 'No radar images available'}), 404
        
        # Blend images
        logger.info(f"Blending {len(weighted_images)} radar images")
        blended_data = RadarBlender.blend_images(weighted_images)
        
        if not blended_data:
            return jsonify({'error': 'Failed to blend images'}), 500
        
        # Cache result
        cache[cache_key] = {
            'time': time.time(),
            'data': blended_data
        }

        logger.info(f"Successfully returned blended radar image ({len(blended_data)} bytes)")
        response = send_file(
            BytesIO(blended_data),
            mimetype='image/png',
            as_attachment=False,
            download_name='radar.png'
        )
        # Add geographic bounds headers for Google Maps compatibility
        response.headers['X-Radar-Bounds-South'] = str(bounds['south'])
        response.headers['X-Radar-Bounds-West'] = str(bounds['west'])
        response.headers['X-Radar-Bounds-North'] = str(bounds['north'])
        response.headers['X-Radar-Bounds-East'] = str(bounds['east'])
        return response
        
    except ValueError as e:
        logger.error(f"Invalid parameters: {e}")
        return jsonify({'error': f'Invalid parameters: {e}'}), 400
    except Exception as e:
        logger.error(f"Error processing radar request: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/providers')
def get_providers():
    """
    Get available radar providers for a location.
    
    Query parameters:
        lat: Latitude
        lng: Longitude
        type: Map type (regional-radar or radar)
    """
    try:
        lat = float(request.args.get('lat', -33.8688))
        lng = float(request.args.get('lng', 151.2093))
        map_type = request.args.get('type', 'regional-radar')
        
        providers = api.get_map_providers(lat, lng, map_type, offset=-30, limit=30)
        
        return jsonify(providers)
        
    except Exception as e:
        logger.error(f"Error getting providers: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/radar/bounds')
def get_radar_bounds():
    """
    Get geographic bounds for a radar view (for Google Maps overlay setup).

    Query parameters:
        lat: Latitude
        lng: Longitude
        zoom: Zoom level

    Returns:
        JSON with bounds: {south, west, north, east}
    """
    try:
        lat = float(request.args.get('lat', -33.8688))
        lng = float(request.args.get('lng', 151.2093))
        zoom = int(request.args.get('zoom', 10))

        # Validate coordinates
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Invalid latitude'}), 400
        if not (-180 <= lng <= 180):
            return jsonify({'error': 'Invalid longitude'}), 400
        if not (1 <= zoom <= 20):
            return jsonify({'error': 'Invalid zoom level'}), 400

        # Calculate zoom radius
        zoom_radius_km = 5000 / (2 ** (zoom - 5))

        # Calculate geographic bounds
        lat_offset = zoom_radius_km / 111.0
        lng_offset = zoom_radius_km / (111.0 * np.cos(np.radians(lat)))

        bounds = {
            'south': lat - lat_offset,
            'west': lng - lng_offset,
            'north': lat + lat_offset,
            'east': lng + lng_offset,
            'center_lat': lat,
            'center_lng': lng,
            'radius_km': zoom_radius_km
        }

        return jsonify(bounds)

    except Exception as e:
        logger.error(f"Error calculating bounds: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/timestamps')
def get_timestamps():
    """
    Get available timestamps for radar imagery.
    
    Query parameters:
        lat: Latitude
        lng: Longitude
        type: Map type
    """
    try:
        lat = float(request.args.get('lat', -33.8688))
        lng = float(request.args.get('lng', 151.2093))
        map_type = request.args.get('type', 'regional-radar')
        
        providers = api.get_map_providers(lat, lng, map_type, offset=-120, limit=120)
        
        if not providers:
            return jsonify([])
        
        # Get all unique timestamps
        timestamps = set()
        for provider in providers:
            for overlay in provider.get('overlays', []):
                timestamps.add(overlay['dateTime'])
        
        return jsonify(sorted(list(timestamps)))
        
    except Exception as e:
        logger.error(f"Error getting timestamps: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/')
def index():
    """Root endpoint."""
    return jsonify({
        'name': 'WillyWeather Radar Addon',
        'version': '1.0.5',
        'status': 'running',
        'endpoints': {
            'radar': '/api/radar?lat={lat}&lng={lng}&zoom={zoom}&timestamp={timestamp}',
            'radar_bounds': '/api/radar/bounds?lat={lat}&lng={lng}&zoom={zoom}',
            'providers': '/api/providers?lat={lat}&lng={lng}&type={type}',
            'timestamps': '/api/timestamps?lat={lat}&lng={lng}&type={type}',
            'health': '/api/health'
        },
        'features': {
            'zoom_threshold': 'National radar for zoom ≤10, regional blending for zoom ≥11',
            'image_smoothing': 'Gaussian blur and unsharp mask for reduced pixelation',
            'google_maps': 'Geographic bounds in response headers (X-Radar-Bounds-*)'
        }
    })


def main():
    """Main entry point."""
    global api
    
    logger.info("=" * 60)
    logger.info("Starting WillyWeather Radar Addon v1.0.5")
    logger.info("=" * 60)
    
    # Load configuration
    load_config()
    
    # Initialize API
    api = WillyWeatherAPI(config['api_key'])
    logger.info("WillyWeather API initialized")
    
    # Start Flask app
    port = int(os.environ.get('PORT', 8099))
    logger.info(f"Starting web server on 0.0.0.0:{port}")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
