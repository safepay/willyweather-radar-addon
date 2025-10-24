# WillyWeather Radar Add-on

![Logo](https://via.placeholder.com/400x200?text=WillyWeather+Radar)

Fetch and serve Australian weather radar imagery from WillyWeather.com.au with intelligent multi-radar blending.

## Features

✨ **Smart Radar Selection** - Automatically switches between regional and national radar based on zoom level

🎨 **Multi-Radar Blending** - Seamlessly combines multiple regional radars for complete coverage

⚡ **Performance** - Built-in caching reduces API calls and improves load times

🔒 **Secure** - Uses Home Assistant's ingress system for secure access

🌏 **Australia-Wide** - Coverage for all of Australia

## Quick Start

1. **Get API Key**: Sign up at [WillyWeather](https://www.willyweather.com.au/info/api.html)
2. **Configure**: Enter your API key in the Configuration tab
3. **Start**: Click Start to launch the add-on
4. **Install Card**: Install the companion [WillyWeather Radar Card](https://github.com/yourusername/willyweather-radar-card)

## Configuration

```yaml
api_key: "your_api_key_here"
cache_duration: 300  # seconds
log_level: info
```

## API Endpoints

The add-on exposes these endpoints via ingress:

- `/api/radar` - Get radar image
- `/api/providers` - Get available providers
- `/api/timestamps` - Get available timestamps
- `/api/health` - Health check

## Support

- [Documentation](https://github.com/yourusername/willyweather-radar-addon)
- [Issues](https://github.com/yourusername/willyweather-radar-addon/issues)
- [Community Forum](https://community.home-assistant.io/)

---

Weather data provided by [WillyWeather](https://www.willyweather.com.au/)
