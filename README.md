# WillyWeather Radar Add-on for Home Assistant

This Home Assistant add-on fetches and serves Australian weather radar imagery from WillyWeather.com.au. It intelligently blends multiple regional radars based on your location and zoom level, or switches to the national radar when viewing larger areas.

## Features

- **Smart Radar Selection**: Automatically chooses between regional and national radar based on zoom level
- **Multi-Radar Blending**: Combines multiple regional radars for seamless coverage
- **Caching**: Reduces API calls and improves performance
- **Ingress Support**: Secure access through Home Assistant's ingress system
- **RESTful API**: Easy integration with custom cards and automations

## Installation

1. Add this repository to your Home Assistant add-on store:
   - Navigate to **Settings** → **Add-ons** → **Add-on Store**
   - Click the three dots menu (⋮) and select **Repositories**
   - Add: `https://github.com/yourusername/willyweather-radar-addon`

2. Install the "WillyWeather Radar" add-on

3. Configure the add-on with your WillyWeather API key

4. Start the add-on

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
- `zoom` (required): Map zoom level (5-15)
- `timestamp` (optional): Specific timestamp (YYYY-MM-DD HH:MM:SS)

**Returns:** PNG image

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
2. **National vs Regional**:
   - **Zoom > 1500km radius** (covering multiple states): Uses national radar
   - **Zoom < 1500km radius**: Uses regional radars with blending

### Multi-Radar Blending

For regional radar views:

1. Queries WillyWeather for nearby radar stations
2. Calculates coverage overlap for each radar with the viewing area
3. Downloads overlays from radars with significant coverage
4. Blends images using weighted averaging based on coverage percentages
5. Returns a seamless composite image

This approach eliminates visible seams between radar coverage areas and provides the best available data for any location.

## Usage with Custom Cards

This add-on is designed to work with the companion custom card. Access the radar images through Home Assistant's ingress system:

```javascript
// In your custom card
const addonUrl = '/api/hassio_ingress/{ingress_token}';
const radarUrl = `${addonUrl}/api/radar?lat=${lat}&lng=${lng}&zoom=${zoom}`;
```

See the [WillyWeather Radar Card](https://github.com/yourusername/willyweather-radar-card) repository for the companion Lovelace card.

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
