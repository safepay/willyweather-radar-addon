#!/usr/bin/env python3
"""WillyWeather Radar Addon for Home Assistant."""

import json
import logging
import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from io import BytesIO
from math import radians, cos, sin, asin, sqrt
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
    response.headers.add('Access-Control-Expose-Headers', 'X-Radar-Bounds-South, X-Radar-Bounds-West, X-Radar-Bounds-North, X-Radar-Bounds-East')
    return response

# Configuration
CONFIG_PATH = '/data/options.json'
CACHE_DIR = '/data/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Global configuration
config = {}
cache = {}
RADAR_STATIONS = None


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


def load_radar_stations():
    """Load radar station data from radars.json"""
    global RADAR_STATIONS
    try:
        radar_file = '/app/radars.json'
        if not os.path.exists(radar_file):
            logger.error(f"Radar stations file not found at {radar_file}")
            RADAR_STATIONS = []
            return
            
        with open(radar_file, 'r') as f:
            data = json.load(f)
            RADAR_STATIONS = data['features']
            logger.info(f"Loaded {len(RADAR_STATIONS)} radar stations")
    except Exception as e:
        logger.error(f"Failed to load radar stations: {e}", exc_info=True)
        RADAR_STATIONS = []


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r


def find_nearby_radars(lat, lng, max_distance_km=500):
    """
    Find radar stations within max_distance_km of the given location.
    Returns list of (station, distance) tuples sorted by distance.
    """
    if not RADAR_STATIONS:
        logger.debug("No radar stations loaded")
        return []
    
    nearby = []
    for station in RADAR_STATIONS:
        coords = station['geometry']['coordinates']
        # GeoJSON format: [longitude, latitude]
        station_lng, station_lat = coords[0], coords[1]
        
        # haversine_distance expects: (lat1, lng1, lat2, lng2)
        distance = haversine_distance(lat, lng, station_lat, station_lng)
        
        if distance <= max_distance_km:
            nearby.append((station, distance))
    
    # Sort by distance (closest first)
    nearby.sort(key=lambda x: x[1])
    
    logger.debug(f"Found {len(nearby)} radars within {max_distance_km}km of ({lat:.2f}, {lng:.2f})")
    if nearby:
        logger.debug(f"  Closest: {nearby[0][0]['properties']['name']} at {nearby[0][1]:.1f}km")
    
    return nearby

def select_radars_for_blending(lat, lng, zoom):
    """
    Select radar(s) for display based on proximity.
    
    Strategy:
    - Always prefer regional radars if any are within 260km
    - Only use national if NO regional radars available
    - Blend 2-3 best radars based on coverage
    """
    zoom_radius_km = 5000 / (2 ** (zoom - 5))
    
    # Find nearby radars within 260km
    nearby = find_nearby_radars(lat, lng, max_distance_km=260)
    
    logger.info(f"Found {len(nearby)} nearby radars within 260km (view radius: {zoom_radius_km:.0f}km)")
    
    if not nearby:
        logger.info("Using national: no nearby radars within 260km")
        return False, [], 'national'
    
    # Calculate coverage for each radar
    radars_with_coverage = []
    
    for station, distance in nearby[:5]:  # Only check first 5 closest
        radar_lat = station['geometry']['coordinates'][1]
        radar_lng = station['geometry']['coordinates'][0]
        radar_name = station['properties']['name']
        
        bounds = {
            'minLat': radar_lat - 2.35,
            'maxLat': radar_lat + 2.35,
            'minLng': radar_lng - 2.35,
            'maxLng': radar_lng + 2.35
        }
        
        coverage = RadarBlender.calculate_coverage(bounds, lat, lng, zoom_radius_km)
        
        # LOWER threshold: 5% instead of 10% to catch more radars at wide zoom
        if coverage >= 0.05:
            radars_with_coverage.append((station, coverage, distance))
            logger.info(f"  ✓ {radar_name} qualifies: coverage={coverage:.1%}, distance={distance:.1f}km")
        else:
            logger.debug(f"  ✗ {radar_name} rejected: coverage={coverage:.1%} (< 5%), distance={distance:.1f}km")
    
    logger.info(f"Total radars qualifying: {len(radars_with_coverage)}")
    
    # If we have nearby radars but none qualify by coverage, USE THEM ANYWAY
    # This prevents losing rain data when zooming out
    if not radars_with_coverage and len(nearby) > 0:
        logger.info("No radars meet coverage threshold, but radars are nearby - using them anyway")
        
        # Use the 2 closest radars regardless of coverage
        for station, distance in nearby[:2]:
            radar_lat = station['geometry']['coordinates'][1]
            radar_lng = station['geometry']['coordinates'][0]
            radar_name = station['properties']['name']
            
            bounds = {
                'minLat': radar_lat - 2.35,
                'maxLat': radar_lat + 2.35,
                'minLng': radar_lng - 2.35,
                'maxLng': radar_lng + 2.35
            }
            
            coverage = RadarBlender.calculate_coverage(bounds, lat, lng, zoom_radius_km)
            radars_with_coverage.append((station, coverage, distance))
            logger.info(f"  ✓ {radar_name} added (fallback): coverage={coverage:.1%}, distance={distance:.1f}km")
    
    # If still no radars, use national
    if not radars_with_coverage:
        logger.info("Using national: no suitable radars found")
        return False, [], 'national'
    
    # Sort by coverage (best first)
    radars_with_coverage.sort(key=lambda x: x[1], reverse=True)
    
    best_radar = radars_with_coverage[0]
    best_coverage = best_radar[1]
    
    # If zoomed in very close (zoom 12+) and one radar covers >90%, use single
    if zoom >= 12 and best_coverage >= 0.90:
        logger.info(f"Using single regional: {best_radar[0]['properties']['name']} "
                   f"with {best_coverage:.1%} coverage at zoom {zoom}")
        return True, [best_radar], 'single'
    
    # Otherwise, always blend 2-3 best radars
    blend_radars = radars_with_coverage[:3]
    radar_names = [r[0]['properties']['name'] for r in blend_radars]
    
    logger.info(f"Using blended regional: {len(blend_radars)} radars "
               f"({', '.join(radar_names)}) at zoom {zoom}")
    
    return True, blend_radars, 'blend'

def get_cache_key(radar_names, timestamp):
    """Generate cache key for blended radar."""
    radar_str = ','.join(sorted(radar_names))
    return f"blend_{radar_str}_{timestamp}".replace(' ', '_').replace(':', '-')


def get_cached_image(cache_key):
    """Get cached blended image if available and not expired."""
    global cache
    
    if cache_key in cache:
        cached_data, cached_time = cache[cache_key]
        age = time.time() - cached_time
        cache_duration = config.get('cache_duration', 300)
        
        if age < cache_duration:
            logger.info(f"✓ Cache hit: {cache_key} (age: {age:.0f}s)")
            return cached_data
        else:
            del cache[cache_key]
            logger.debug(f"Cache expired: {cache_key}")
    
    # Check file cache
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < config.get('cache_duration', 300):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"✓ File cache hit: {cache_key}")
                cache[cache_key] = (cached_data, time.time())
                return cached_data
            else:
                os.remove(cache_file)
        except Exception as e:
            logger.error(f"Error reading cache file: {e}")
    
    return None

def save_to_cache(cache_key, data):
    """Save blended image to cache (memory and disk)."""
    global cache
    
    cache[cache_key] = (data, time.time())
    
    try:
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✓ Cached: {cache_key} (mem + disk)")
    except Exception as e:
        logger.error(f"Error writing cache file: {e}")
    
    # Clean old memory cache entries
    if len(cache) > 20:
        oldest_keys = sorted(cache.keys(), key=lambda k: cache[k][1])[:len(cache) - 20]
        for key in oldest_keys:
            del cache[key]


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
    def blend_images(images: List[Tuple[bytes, float, Dict]], target_bounds: Dict, 
                     target_size: Tuple[int, int] = (512, 512),  # REDUCED from 1024
                     smooth: bool = False) -> Optional[bytes]:  # DISABLED smoothing by default
        """
        Memory-efficient radar image blending.
        
        Args:
            images: List of (image_bytes, weight, bounds_dict) tuples
            target_bounds: Output bounds
            target_size: Output size (smaller = less memory)
            smooth: Apply smoothing (costs memory)
        
        Returns:
            Blended image bytes or None
        """
        if not images:
            return None
    
        try:
            logger.info(f"Blending {len(images)} images at {target_size[0]}x{target_size[1]}")
            
            canvas_array = np.zeros((target_size[1], target_size[0], 4), dtype=np.float32)  # Changed from uint16
            total_weight_array = np.zeros((target_size[1], target_size[0]), dtype=np.float32)

            target_lat_range = target_bounds['maxLat'] - target_bounds['minLat']
            target_lng_range = target_bounds['maxLng'] - target_bounds['minLng']
    
            for idx, (img_bytes, weight, img_bounds) in enumerate(images):
                logger.debug(f"Processing radar {idx+1}/{len(images)}")
                
                # Load radar image
                radar_img = Image.open(BytesIO(img_bytes)).convert('RGBA')
                radar_array = np.array(radar_img, dtype=np.uint8)  # Keep as uint8
                
                # Calculate overlap region
                radar_lat_min = img_bounds['minLat']
                radar_lat_max = img_bounds['maxLat']
                radar_lng_min = img_bounds['minLng']
                radar_lng_max = img_bounds['maxLng']
                
                overlap_lat_min = max(target_bounds['minLat'], radar_lat_min)
                overlap_lat_max = min(target_bounds['maxLat'], radar_lat_max)
                overlap_lng_min = max(target_bounds['minLng'], radar_lng_min)
                overlap_lng_max = min(target_bounds['maxLng'], radar_lng_max)
                
                if overlap_lat_min >= overlap_lat_max or overlap_lng_min >= overlap_lng_max:
                    logger.debug(f"Radar {idx+1}: no overlap, skipping")
                    continue
                
                # Calculate pixel coordinates in target canvas
                canvas_x_start = int((overlap_lng_min - target_bounds['minLng']) / target_lng_range * target_size[0])
                canvas_x_end = int((overlap_lng_max - target_bounds['minLng']) / target_lng_range * target_size[0])
                canvas_y_start = int((target_bounds['maxLat'] - overlap_lat_max) / target_lat_range * target_size[1])
                canvas_y_end = int((target_bounds['maxLat'] - overlap_lat_min) / target_lat_range * target_size[1])
                
                # Calculate source coordinates
                radar_lat_range = radar_lat_max - radar_lat_min
                radar_lng_range = radar_lng_max - radar_lng_min
                radar_height, radar_width = radar_array.shape[:2]
                
                src_x_start = int((overlap_lng_min - radar_lng_min) / radar_lng_range * radar_width)
                src_x_end = int((overlap_lng_max - radar_lng_min) / radar_lng_range * radar_width)
                src_y_start = int((radar_lat_max - overlap_lat_max) / radar_lat_range * radar_height)
                src_y_end = int((radar_lat_max - overlap_lat_min) / radar_lat_range * radar_height)
                
                # Bounds checking
                src_x_start = max(0, min(src_x_start, radar_width - 1))
                src_x_end = max(0, min(src_x_end, radar_width))
                src_y_start = max(0, min(src_y_start, radar_height - 1))
                src_y_end = max(0, min(src_y_end, radar_height))
                
                if src_x_end <= src_x_start or src_y_end <= src_y_start:
                    continue
                
                # Extract and resize
                radar_crop = radar_array[src_y_start:src_y_end, src_x_start:src_x_end]
                
                canvas_width = canvas_x_end - canvas_x_start
                canvas_height = canvas_y_end - canvas_y_start
                
                if canvas_width <= 0 or canvas_height <= 0:
                    continue
                
                radar_crop_img = Image.fromarray(radar_crop, mode='RGBA')
                # Use LANCZOS for better quality at lower resolution
                radar_crop_resized = radar_crop_img.resize((canvas_width, canvas_height), Image.LANCZOS)
                radar_crop_array = np.array(radar_crop_resized, dtype=np.uint8)
                
                # Bounds checking for canvas
                canvas_y_start = max(0, min(canvas_y_start, target_size[1]))
                canvas_y_end = max(0, min(canvas_y_end, target_size[1]))
                canvas_x_start = max(0, min(canvas_x_start, target_size[0]))
                canvas_x_end = max(0, min(canvas_x_end, target_size[0]))
                
                if canvas_y_end <= canvas_y_start or canvas_x_end <= canvas_x_start:
                    continue
                
                # Adjust crop size
                actual_height = canvas_y_end - canvas_y_start
                actual_width = canvas_x_end - canvas_x_start
                radar_crop_array = radar_crop_array[:actual_height, :actual_width]
                
                # Weight by alpha channel
                alpha_channel = radar_crop_array[:, :, 3].astype(np.float32) / 255.0
                pixel_weight = alpha_channel * weight
                
                # Accumulate with uint16 to prevent overflow
                canvas_array[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] += \
                    radar_crop_array.astype(np.uint16) * pixel_weight[:, :, np.newaxis]
                total_weight_array[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] += pixel_weight
                
                logger.debug(f"Radar {idx+1}: added to canvas")
    
            # Normalize by total weight
            mask = total_weight_array > 0
            for i in range(4):
                canvas_array[:, :, i][mask] = (canvas_array[:, :, i][mask] / total_weight_array[mask]).astype(np.uint16)
    
            # Convert to uint8
            canvas_array = np.clip(canvas_array, 0, 255).astype(np.uint8)
            result_img = Image.fromarray(canvas_array, mode='RGBA')
    
            # Optional smoothing (costs memory, disabled by default)
            if smooth:
                result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
            # Save with compression
            output = BytesIO()
            result_img.save(output, format='PNG', optimize=True, compress_level=6)
            
            result_size = len(output.getvalue())
            logger.info(f"Blending complete: output size {result_size} bytes")
            
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
        'version': '1.0.6',
        'api_configured': bool(config.get('api_key')),
        'radar_stations_loaded': len(RADAR_STATIONS) if RADAR_STATIONS else 0
    })


@app.route('/api/radar/info')
def get_radar_info():
    """
    Get information about which radar type will be used and why.
    Useful for debugging and the card UI.
    """
    try:
        lat = float(request.args.get('lat', -33.8688))
        lng = float(request.args.get('lng', 151.2093))
        zoom = int(request.args.get('zoom', 10))

        use_regional, nearby_stations = should_use_regional_radar(lat, lng, zoom)
        zoom_radius_km = 5000 / (2 ** (zoom - 5))

        nearby_radars = find_nearby_radars(lat, lng, max_distance_km=500)
        
        return jsonify({
            'use_regional': use_regional,
            'map_type': 'regional-radar' if use_regional else 'radar',
            'view_radius_km': round(zoom_radius_km, 1),
            'nearby_radars': [
                {
                    'name': station['properties']['name'],
                    'id': station['properties']['id'],
                    'distance_km': round(dist, 1)
                }
                for station, dist in nearby_radars[:10]
            ],
            'radars_to_use': [
                station['properties']['name'] for station in nearby_stations
            ]
        })
        
    except Exception as e:
        logger.error(f"Error getting radar info: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/radar')
def get_radar():
    """Get radar imagery - with caching."""
    try:
        lat = float(request.args.get('lat', -33.8688))
        lng = float(request.args.get('lng', 151.2093))
        zoom = int(request.args.get('zoom', 10))
        timestamp = request.args.get('timestamp')
        
        zoom_radius_km = 5000 / (2 ** (zoom - 5))
        
        use_regional, radars, mode = select_radars_for_blending(lat, lng, zoom)
        
        # === NATIONAL RADAR ===
        if not use_regional:
            logger.info(f"Requesting national radar")
            providers = api.get_map_providers(lat, lng, 'radar', offset=-120, limit=120)
            
            if not providers:
                return jsonify({'error': 'No radar data available'}), 404
            
            provider = providers[0]
            overlays = provider.get('overlays', [])
            
            if not overlays:
                return jsonify({'error': 'No overlays available'}), 404
            
            overlay = next((o for o in overlays if o['dateTime'] == timestamp), overlays[-1]) if timestamp else overlays[-1]
            
            logger.info(f"Using national radar, timestamp {overlay['dateTime']}")
            
            image_data = api.download_overlay(provider['overlayPath'], overlay['name'])
            
            if not image_data:
                return jsonify({'error': 'Failed to download'}), 500
            
            response = send_file(BytesIO(image_data), mimetype='image/png', as_attachment=False)
            response.headers['X-Radar-Bounds-South'] = str(provider['bounds']['minLat'])
            response.headers['X-Radar-Bounds-West'] = str(provider['bounds']['minLng'])
            response.headers['X-Radar-Bounds-North'] = str(provider['bounds']['maxLat'])
            response.headers['X-Radar-Bounds-East'] = str(provider['bounds']['maxLng'])
            
            logger.info(f"Successfully returned national radar ({len(image_data)} bytes)")
            return response
        
        # === SINGLE REGIONAL RADAR ===
        if mode == 'single':
            radar_station = radars[0][0]
            radar_lat = radar_station['geometry']['coordinates'][1]
            radar_lng = radar_station['geometry']['coordinates'][0]
            
            logger.info(f"Requesting single radar at ({radar_lat:.4f}, {radar_lng:.4f})")
            providers = api.get_map_providers(radar_lat, radar_lng, 'regional-radar', offset=-120, limit=120)
            
            if not providers:
                return jsonify({'error': 'No radar data'}), 404
            
            provider = providers[0]
            overlays = provider.get('overlays', [])
            
            if not overlays:
                return jsonify({'error': 'No overlays'}), 404
            
            overlay = next((o for o in overlays if o['dateTime'] == timestamp), overlays[-1]) if timestamp else overlays[-1]
            
            logger.info(f"Using {provider['name']}, timestamp {overlay['dateTime']}")
            
            image_data = api.download_overlay(provider['overlayPath'], overlay['name'])
            
            if not image_data:
                return jsonify({'error': 'Failed to download'}), 500
            
            response = send_file(BytesIO(image_data), mimetype='image/png', as_attachment=False)
            response.headers['X-Radar-Bounds-South'] = str(provider['bounds']['minLat'])
            response.headers['X-Radar-Bounds-West'] = str(provider['bounds']['minLng'])
            response.headers['X-Radar-Bounds-North'] = str(provider['bounds']['maxLat'])
            response.headers['X-Radar-Bounds-East'] = str(provider['bounds']['maxLng'])
            
            logger.info(f"Successfully returned single radar ({len(image_data)} bytes)")
            return response
        
        # === BLENDED REGIONAL RADAR ===
        logger.info(f"Blending {len(radars)} radars")
        
        # Generate cache key
        radar_names = [r[0]['properties']['name'] for r in radars]
        cache_key = get_cache_key(radar_names, timestamp or 'latest')
        
        # Check cache
        cached = get_cached_image(cache_key)
        if cached:
            composite_bounds, blended_image = cached
            
            response = send_file(BytesIO(blended_image), mimetype='image/png', as_attachment=False)
            response.headers['X-Radar-Bounds-South'] = str(composite_bounds['minLat'])
            response.headers['X-Radar-Bounds-West'] = str(composite_bounds['minLng'])
            response.headers['X-Radar-Bounds-North'] = str(composite_bounds['maxLat'])
            response.headers['X-Radar-Bounds-East'] = str(composite_bounds['maxLng'])
            response.headers['X-Cache'] = 'HIT'
            
            logger.info(f"Returned cached blend ({len(blended_image)} bytes)")
            return response
        
        logger.info("Cache miss - blending")
        
        # ... rest of blending code continues ...        
        # Generate cache key
        radar_names = [r[0]['properties']['name'] for r in radars]
        cache_key = get_cache_key(radar_names, timestamp or 'latest')
        
        # Check cache
        cached = get_cached_image(cache_key)
        if cached:
            composite_bounds, blended_image = cached
            
            response = send_file(BytesIO(blended_image), mimetype='image/png', as_attachment=False)
            response.headers['X-Radar-Bounds-South'] = str(composite_bounds['minLat'])
            response.headers['X-Radar-Bounds-West'] = str(composite_bounds['minLng'])
            response.headers['X-Radar-Bounds-North'] = str(composite_bounds['maxLat'])
            response.headers['X-Radar-Bounds-East'] = str(composite_bounds['maxLng'])
            response.headers['X-Cache'] = 'HIT'
            
            logger.info(f"Returned cached blend ({len(blended_image)} bytes)")
            return response
        
        logger.info("Cache miss - blending")
        
        # Fetch and blend (existing code)
        weighted_images = []
        all_bounds = []
        
        for radar_station, coverage, distance in radars:
            radar_lat = radar_station['geometry']['coordinates'][1]
            radar_lng = radar_station['geometry']['coordinates'][0]
            radar_name = radar_station['properties']['name']
            
            providers = api.get_map_providers(radar_lat, radar_lng, 'regional-radar', offset=-120, limit=120)
            
            if not providers:
                logger.warning(f"No provider for {radar_name}")
                continue
            
            provider = providers[0]
            overlays = provider.get('overlays', [])
            
            if not overlays:
                logger.warning(f"No overlays for {radar_name}")
                continue
            
            overlay = None
            if timestamp:
                overlay = next((o for o in overlays if o['dateTime'] == timestamp), None)
                if not overlay:
                    logger.warning(f"{radar_name} missing timestamp {timestamp}")
                    continue
            else:
                overlay = overlays[-1]
            
            image_data = api.download_overlay(provider['overlayPath'], overlay['name'])
            
            if image_data:
                weighted_images.append((image_data, coverage, provider['bounds']))
                all_bounds.append(provider['bounds'])
                logger.info(f"✓ Added {radar_name}")
            else:
                logger.warning(f"Failed to download {radar_name}")
        
        if not weighted_images:
            return jsonify({'error': 'No radar images available'}), 404
        
        composite_bounds = {
            'minLat': min(b['minLat'] for b in all_bounds),
            'maxLat': max(b['maxLat'] for b in all_bounds),
            'minLng': min(b['minLng'] for b in all_bounds),
            'maxLng': max(b['maxLng'] for b in all_bounds)
        }
        
        blended_image = RadarBlender.blend_images(
            weighted_images,
            composite_bounds,
            target_size=(512, 512),
            smooth=False
        )
        
        if not blended_image:
            return jsonify({'error': 'Failed to blend'}), 500
        
        # Save to cache
        save_to_cache(cache_key, (composite_bounds, blended_image))
        
        response = send_file(BytesIO(blended_image), mimetype='image/png', as_attachment=False)
        response.headers['X-Radar-Bounds-South'] = str(composite_bounds['minLat'])
        response.headers['X-Radar-Bounds-West'] = str(composite_bounds['minLng'])
        response.headers['X-Radar-Bounds-North'] = str(composite_bounds['maxLat'])
        response.headers['X-Radar-Bounds-East'] = str(composite_bounds['maxLng'])
        response.headers['X-Cache'] = 'MISS'
        
        logger.info(f"Returned fresh blend ({len(blended_image)} bytes)")
        return response
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
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
    """Get timestamps - find common timestamps when blending."""
    try:
        lat = float(request.args.get('lat', -33.8688))
        lng = float(request.args.get('lng', 151.2093))
        zoom = int(request.args.get('zoom', 10))
        
        # Use same selection logic
        use_regional, radars, mode = select_radars_for_blending(lat, lng, zoom)
        
        # === NATIONAL RADAR ===
        if not use_regional:
            providers = api.get_map_providers(lat, lng, 'radar', offset=-120, limit=120)
            
            if not providers:
                return jsonify([])
            
            timestamps = set()
            for overlay in providers[0].get('overlays', []):
                timestamps.add(overlay['dateTime'])
            
            return jsonify(sorted(list(timestamps)))
        
        # === SINGLE REGIONAL RADAR ===
        if mode == 'single':
            radar_station = radars[0][0]
            radar_lat = radar_station['geometry']['coordinates'][1]
            radar_lng = radar_station['geometry']['coordinates'][0]
            
            providers = api.get_map_providers(radar_lat, radar_lng, 'regional-radar', offset=-120, limit=120)
            
            if not providers:
                return jsonify([])
            
            timestamps = set()
            for overlay in providers[0].get('overlays', []):
                timestamps.add(overlay['dateTime'])
            
            logger.info(f"Returning {len(timestamps)} timestamps from {providers[0]['name']}")
            return jsonify(sorted(list(timestamps)))
        
        # === BLENDED REGIONAL RADAR ===
        # Find common timestamps across all radars
        all_timestamp_sets = []
        
        for radar_station, coverage, distance in radars:
            radar_lat = radar_station['geometry']['coordinates'][1]
            radar_lng = radar_station['geometry']['coordinates'][0]
            radar_name = radar_station['properties']['name']
            
            providers = api.get_map_providers(radar_lat, radar_lng, 'regional-radar', offset=-120, limit=120)
            
            if providers:
                timestamps = set()
                for overlay in providers[0].get('overlays', []):
                    timestamps.add(overlay['dateTime'])
                all_timestamp_sets.append((radar_name, timestamps))
                logger.debug(f"{radar_name}: {len(timestamps)} timestamps")
        
        if not all_timestamp_sets:
            return jsonify([])
        
        # Find intersection (common timestamps)
        common_timestamps = all_timestamp_sets[0][1]
        for radar_name, timestamps in all_timestamp_sets[1:]:
            before = len(common_timestamps)
            common_timestamps = common_timestamps.intersection(timestamps)
            logger.debug(f"After {radar_name}: {len(common_timestamps)} common ({before - len(common_timestamps)} removed)")
        
        logger.info(f"Returning {len(common_timestamps)} common timestamps across {len(all_timestamp_sets)} radars")
        
        return jsonify(sorted(list(common_timestamps)))
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify([])
        
@app.route('/')
def index():
    """Root endpoint."""
    return jsonify({
        'name': 'WillyWeather Radar Addon',
        'version': '1.0.6',
        'status': 'running',
        'radar_stations': len(RADAR_STATIONS) if RADAR_STATIONS else 0,
        'endpoints': {
            'radar': '/api/radar?lat={lat}&lng={lng}&zoom={zoom}&timestamp={timestamp}',
            'radar_bounds': '/api/radar/bounds?lat={lat}&lng={lng}&zoom={zoom}',
            'radar_info': '/api/radar/info?lat={lat}&lng={lng}&zoom={zoom}',
            'providers': '/api/providers?lat={lat}&lng={lng}&type={type}',
            'timestamps': '/api/timestamps?lat={lat}&lng={lng}&zoom={zoom}',
            'health': '/api/health'
        },
        'features': {
            'location_aware_radar_selection': 'Automatically chooses regional or national based on radar proximity',
            'image_smoothing': 'Gaussian blur and unsharp mask for reduced pixelation',
            'google_maps': 'Geographic bounds in response headers (X-Radar-Bounds-*)'
        }
    })

@app.route('/api/test/debug-radar-selection')
def test_debug_radar_selection():
    """Debug why radars aren't being selected for blending."""
    lat, lng, zoom = -37.103206, 144.116305, 8
    
    zoom_radius_km = 5000 / (2 ** (zoom - 5))
    
    # Find nearby radars
    nearby = find_nearby_radars(lat, lng, max_distance_km=500)
    
    result = {
        'location': f'lat={lat}, lng={lng}, zoom={zoom}',
        'zoom_radius_km': zoom_radius_km,
        'nearby_radars': []
    }
    
    for station, distance in nearby[:10]:
        radar_lat = station['geometry']['coordinates'][1]
        radar_lng = station['geometry']['coordinates'][0]
        radar_name = station['properties']['name']
        
        bounds = {
            'minLat': radar_lat - 2.35,
            'maxLat': radar_lat + 2.35,
            'minLng': radar_lng - 2.35,
            'maxLng': radar_lng + 2.35
        }
        
        coverage = RadarBlender.calculate_coverage(bounds, lat, lng, zoom_radius_km)
        qualifies = coverage >= 0.10
        
        result['nearby_radars'].append({
            'name': radar_name,
            'distance_km': round(distance, 1),
            'coverage_pct': f"{coverage*100:.1f}%",
            'qualifies_10pct': 'yes' if qualifies else 'no',  # Changed to string
            'coordinates': [radar_lng, radar_lat]
        })
    
    return jsonify(result)
    
def main():
    """Main entry point."""
    global api
    
    logger.info("=" * 60)
    logger.info("Starting WillyWeather Radar Addon v1.0.6")
    logger.info("=" * 60)
    
    # Load configuration
    load_config()
    
    # Load radar stations
    logger.info("Loading radar stations database...")
    load_radar_stations()
    
    # Verify radar stations loaded
    if not RADAR_STATIONS or len(RADAR_STATIONS) == 0:
        logger.error("=" * 60)
        logger.error("❌ CRITICAL: No radar stations loaded!")
        logger.error("   Location-aware radar selection will NOT work")
        logger.error("   All requests will fall back to national radar")
        logger.error("=" * 60)
    else:
        logger.info(f"✓ Radar station database ready with {len(RADAR_STATIONS)} stations")
        
        # Test the distance calculation with Melbourne
        test_nearby = find_nearby_radars(-37.8136, 144.9631, max_distance_km=200)
        if test_nearby:
            logger.info(f"✓ Distance calculation verified - found {len(test_nearby)} radars near Melbourne")
        else:
            logger.warning("⚠ Distance calculation may have issues - no radars found near Melbourne")
    
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
