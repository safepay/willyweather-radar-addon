# WillyWeather Radar Add-on for Home Assistant

This Home Assistant add-on fetches and serves Australian weather radar imagery from WillyWeather.com.au. It intelligently blends multiple regional radars based on your location and zoom level, or switches to the national radar when viewing larger areas.

## Features

- **Smart Radar Selection**: Automatically chooses between regional and national radar based on zoom level (switches at zoom 10)
- **Multi-Radar Blending**: Combines multiple regional radars for seamless coverage with weighted averaging
- **Image Smoothing**: Advanced smoothing algorithms reduce pixelation for clearer radar images
- **Google Maps Compatible**: Geographic bounds provided in response headers for easy map overlay integration
- **Caching**: Reduces API calls and improves performance
- **Ingress Support**: Secure access through Home Assistant's ingress system
- **RESTful API**: Easy integration with custom cards and automations

## Installation

### Method 1: Add as Repository (Recommended)

1. Add this repository to your Home Assistant add-on store:
   - Navigate to **Settings** → **Add-ons** → **Add-on Store**
   - Click the three dots menu (⋮) in the top right
   - Select **Repositories**
   - Add: `https://github.com/yourusername/willyweather-radar-addon`
   - Click **Add**

2. Refresh the add-on store (you may need to close and reopen it)

3. Find "WillyWeather Radar" in the add-on list

4. Click on it and click **Install**

5. Once installed, go to the **Configuration** tab and add your WillyWeather API key

6. Start the add-on

### Method 2: Manual Installation (Development)

If you're developing or testing:

1. Clone this repository
2. Copy the entire directory to `/addons/willyweather-radar/` on your Home Assistant instance
3. Restart the Supervisor
4. The add-on will appear in your local add-ons list

## Configuration

```yaml
api_key: "your_willyweather_api_key_here"
cache_duration: 300  # Cache duration in seconds (default: 5 minutes)
log_level: info      # Log level: debug, info, warning, error
```

### Getting a WillyWeather API Key

1. Visit [WillyWeather Developer Portal](https://www.willyweather.com.au/info/api.html)
2. Sign up for a free API key
3. Copy your API key and paste it into the add-on configuration

## API Endpoints

Once installed, the add-on exposes the following API endpoints via ingress:

### Get Radar Image

```
GET /api/radar?lat={latitude}&lng={longitude}&zoom={zoom}&timestamp={timestamp}
```

**Parameters:**
- `lat` (required): Latitude (-90 to 90)
- `lng` (required): Longitude (-180 to 180)
- `zoom` (required): Map zoom level (1-20)
  - Zoom ≤10: National radar coverage (~160km+ radius)
  - Zoom ≥11: Regional radar blending (~160km- radius)
- `timestamp` (optional): Specific timestamp (YYYY-MM-DD HH:MM:SS)

**Returns:** PNG image with geographic bounds in response headers:
- `X-Radar-Bounds-South`: Southern latitude boundary
- `X-Radar-Bounds-West`: Western longitude boundary
- `X-Radar-Bounds-North`: Northern latitude boundary
- `X-Radar-Bounds-East`: Eastern longitude boundary

### Get Radar Bounds

```
GET /api/radar/bounds?lat={latitude}&lng={longitude}&zoom={zoom}
```

**Parameters:**
- `lat` (required): Latitude (-90 to 90)
- `lng` (required): Longitude (-180 to 180)
- `zoom` (required): Map zoom level (1-20)

**Returns:** JSON object with geographic bounds:
```json
{
  "south": -34.268,
  "west": 149.809,
  "north": -33.468,
  "east": 152.609,
  "center_lat": -33.8688,
  "center_lng": 151.2093,
  "radius_km": 156.25
}
```

### Get Providers

```
GET /api/providers?lat={latitude}&lng={longitude}&type={type}
```

**Parameters:**
- `lat` (required): Latitude
- `lng` (required): Longitude
- `type` (optional): Map type (regional-radar or radar)

**Returns:** JSON array of radar providers

### Get Timestamps

```
GET /api/timestamps?lat={latitude}&lng={longitude}&type={type}
```

**Parameters:**
- `lat` (required): Latitude
- `lng` (required): Longitude
- `type` (optional): Map type

**Returns:** JSON array of available timestamps

### Health Check

```
GET /api/health
```

**Returns:** `{"status": "ok"}`

## How It Works

### Radar Selection Logic

The add-on uses intelligent logic to determine which radar data to serve:

1. **Zoom Level Analysis**: Calculates the approximate viewing radius based on the zoom level
   - Formula: `zoom_radius_km = 5000 / (2 ^ (zoom - 5))`
   - Examples: Zoom 10 = 156km, Zoom 11 = 78km, Zoom 13 = 39km

2. **National vs Regional**:
   - **Zoom ≤10 (>160km radius)**: Uses national radar (single source)
   - **Zoom ≥11 (≤160km radius)**: Uses regional radars with intelligent blending

### Multi-Radar Blending

For regional radar views (zoom ≥11):

1. Queries WillyWeather for up to 5 nearby radar stations
2. Calculates geographic coverage overlap for each radar with the viewing area
3. Downloads overlays from radars with >5% coverage
4. Blends images using weighted averaging based on coverage percentages
5. Applies image smoothing filters for improved visual quality
6. Returns a seamless composite image

### Image Quality Enhancements

All radar images are processed with advanced smoothing algorithms:

- **BICUBIC Interpolation**: High-quality image resizing for smoother results
- **Gaussian Blur** (radius 0.8): Smooths pixel transitions and reduces artifacts
- **Unsharp Mask** (radius 1.0, 50%): Preserves detail and sharpness after smoothing
- **PNG Optimization**: Reduces file size without quality loss

This approach eliminates visible seams between radar coverage areas, reduces pixelation, and provides the best available data for any location.

## Usage with Custom Cards

This add-on is designed to work with the companion custom card. Access the radar images through Home Assistant's ingress system:

```javascript
// In your custom card
const addonUrl = '/api/hassio_ingress/{ingress_token}';
const radarUrl = `${addonUrl}/api/radar?lat=${lat}&lng=${lng}&zoom=${zoom}`;
```

See the [WillyWeather Radar Card](https://github.com/yourusername/willyweather-radar-card) repository for the companion Lovelace card.

## Google Maps Integration

The addon provides geographic bounds in response headers for easy overlay integration with Google Maps:

### Basic Overlay Example

```javascript
// Fetch radar image with bounds
const lat = -33.8688;
const lng = 151.2093;
const zoom = 12;

const response = await fetch(`/api/radar?lat=${lat}&lng=${lng}&zoom=${zoom}`);
const imageBlob = await response.blob();

// Extract bounds from response headers
const bounds = {
  south: parseFloat(response.headers.get('X-Radar-Bounds-South')),
  west: parseFloat(response.headers.get('X-Radar-Bounds-West')),
  north: parseFloat(response.headers.get('X-Radar-Bounds-North')),
  east: parseFloat(response.headers.get('X-Radar-Bounds-East'))
};

// Create Google Maps ground overlay
const overlay = new google.maps.GroundOverlay(
  URL.createObjectURL(imageBlob),
  bounds,
  { opacity: 0.7 }
);

overlay.setMap(map);
```

### Using the Bounds Endpoint

For pre-calculating bounds before fetching the image:

```javascript
// Get bounds first
const boundsResponse = await fetch(`/api/radar/bounds?lat=${lat}&lng=${lng}&zoom=${zoom}`);
const boundsData = await boundsResponse.json();

console.log(`Viewing radius: ${boundsData.radius_km} km`);
console.log(`Bounds:`, boundsData);

// Then fetch the radar image
const radarResponse = await fetch(`/api/radar?lat=${lat}&lng=${lng}&zoom=${zoom}`);
const imageBlob = await radarResponse.blob();
```

### Dynamic Updates

```javascript
// Update radar overlay when map view changes
map.addListener('bounds_changed', async () => {
  const center = map.getCenter();
  const zoom = map.getZoom();

  const response = await fetch(
    `/api/radar?lat=${center.lat()}&lng=${center.lng()}&zoom=${zoom}`
  );

  const imageBlob = await response.blob();
  const bounds = {
    south: parseFloat(response.headers.get('X-Radar-Bounds-South')),
    west: parseFloat(response.headers.get('X-Radar-Bounds-West')),
    north: parseFloat(response.headers.get('X-Radar-Bounds-North')),
    east: parseFloat(response.headers.get('X-Radar-Bounds-East'))
  };

  // Remove old overlay if exists
  if (radarOverlay) radarOverlay.setMap(null);

  // Add new overlay
  radarOverlay = new google.maps.GroundOverlay(
    URL.createObjectURL(imageBlob),
    bounds,
    { opacity: 0.7 }
  );
  radarOverlay.setMap(map);
});
```

## Troubleshooting

### No Radar Data

- Verify your API key is correct
- Check that you're querying coordinates within Australia
- Review logs: **Settings** → **Add-ons** → **WillyWeather Radar** → **Log**

### Slow Performance

- Increase `cache_duration` in configuration
- Check your internet connection
- Verify WillyWeather API is accessible

### API Rate Limits

- The free WillyWeather API has rate limits
- Use caching to minimize API calls
- Consider upgrading to a paid plan for higher limits

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/yourusername/willyweather-radar-addon/issues) page.

## License

MIT License - See LICENSE file for details

## Credits

- Weather data provided by [WillyWeather](https://www.willyweather.com.au/)
- Developed for the Home Assistant community
