# WillyWeather Radar Add-on Repository

Home Assistant add-on repository for Australian weather radar from WillyWeather.

## Add-ons in this Repository

- **WillyWeather Radar** - Fetch and serve Australian weather radar imagery with intelligent multi-radar blending

## Installation

Add this repository to your Home Assistant instance:

1. Navigate to **Settings** → **Add-ons** → **Add-on Store**
2. Click the **⋮** menu (three dots) in the top right
3. Select **Repositories**
4. Add this URL:
   ```
   https://github.com/safepay/willyweather-radar-addon
   ```
5. Click **Add**
6. Close and reopen the Add-on Store
7. You should see "WillyWeather Radar" in the list

## Add-ons

### WillyWeather Radar

Fetches and serves Australian weather radar imagery from WillyWeather.com.au with intelligent multi-radar blending.

**Features:**
- Smart radar selection (regional vs national)
- Multi-radar blending for seamless coverage
- Built-in caching
- Ingress support for secure access
- RESTful API

**Requirements:**
- WillyWeather API key (free tier available)

See [willyweather-radar/README.md](willyweather-radar/README.md) for detailed documentation.

## Support

- [Issues](https://github.com/safepay/willyweather-radar-addon/issues)
- [Home Assistant Community](https://community.home-assistant.io/)

## License

MIT License - see individual add-on directories for details
