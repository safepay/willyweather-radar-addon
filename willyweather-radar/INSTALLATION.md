# Installation Guide

## Prerequisites

- Home Assistant OS, Supervised, or Container installation
- WillyWeather API key ([Get one here](https://www.willyweather.com.au/info/api.html))

## Step-by-Step Installation

### 1. Add the Repository

1. Navigate to **Settings** → **Add-ons** → **Add-on Store**
2. Click the three dots menu (⋮) in the top right
3. Select **Repositories**
4. Add the repository URL:
   ```
   https://github.com/yourusername/willyweather-radar-addon
   ```
5. Click **Add**

### 2. Install the Add-on

1. Refresh the add-on store page
2. Find "WillyWeather Radar" in the list
3. Click on it
4. Click **Install**
5. Wait for the installation to complete

### 3. Configure the Add-on

1. Go to the **Configuration** tab
2. Enter your WillyWeather API key:
   ```yaml
   api_key: "your_api_key_here"
   cache_duration: 300
   log_level: info
   ```
3. Click **Save**

### 4. Start the Add-on

1. Go to the **Info** tab
2. Enable **Start on boot** (recommended)
3. Enable **Auto update** (recommended)
4. Click **Start**
5. Check the **Log** tab to verify it started successfully

### 5. Verify Installation

1. The add-on should show as "Running" in the Info tab
2. Check logs for any errors
3. You can access the add-on via the **Open Web UI** button (if configured)

## Getting a WillyWeather API Key

1. Visit https://www.willyweather.com.au/info/api.html
2. Click "Register" or "Sign In"
3. Complete the registration form
4. Navigate to your API dashboard
5. Copy your API key
6. Paste it into the add-on configuration

### API Key Limitations

The free tier includes:
- 1,000 API calls per day
- Rate limit: 10 calls per minute

If you need more:
- Consider upgrading to a paid plan
- Increase `cache_duration` to reduce API calls

## Configuration Options

### api_key (required)
Your WillyWeather API key.

**Example:**
```yaml
api_key: "abc123def456"
```

### cache_duration (optional)
Duration in seconds to cache radar images.

**Default:** 300 (5 minutes)

**Example:**
```yaml
cache_duration: 600  # Cache for 10 minutes
```

**Tip:** Increase this value to reduce API calls.

### log_level (optional)
Logging verbosity level.

**Options:** debug, info, warning, error

**Default:** info

**Example:**
```yaml
log_level: debug  # For troubleshooting
```

## Troubleshooting

### Add-on Won't Start

**Check logs:**
1. Go to the add-on page
2. Click the **Log** tab
3. Look for error messages

**Common issues:**
- Invalid API key → Verify your API key
- Network connectivity → Check internet connection
- Port conflict → Ensure port 8099 is available

### API Key Errors

If you see "API key not configured" errors:

1. Double-check your API key in the configuration
2. Make sure there are no extra spaces
3. Verify the key is active in your WillyWeather account
4. Try regenerating the key

### Connection Issues

If the add-on can't reach WillyWeather:

1. Check your internet connection
2. Verify firewall settings aren't blocking outbound connections
3. Check Home Assistant's network configuration
4. Try restarting the add-on

### High Memory Usage

If the add-on uses too much memory:

1. Reduce `cache_duration`
2. Restart the add-on periodically
3. Check for memory leaks in logs
4. Consider upgrading your system's RAM

## Updating

### Automatic Updates

If you enabled "Auto update":
- The add-on will update automatically when new versions are released
- Check the changelog after updates

### Manual Updates

1. Go to the add-on page
2. If an update is available, you'll see an **Update** button
3. Click **Update**
4. Wait for the update to complete
5. The add-on will restart automatically

## Uninstallation

To remove the add-on:

1. Stop the add-on
2. Click **Uninstall**
3. Confirm the uninstallation
4. Remove the repository (optional):
   - Go to **Add-on Store**
   - Click three dots menu → **Repositories**
   - Remove the repository URL

## Next Steps

After installation:

1. Install the [WillyWeather Radar Card](https://github.com/yourusername/willyweather-radar-card)
2. Add the card to your dashboard
3. Configure card settings to match your location

## Support

For help with installation:

- [GitHub Issues](https://github.com/yourusername/willyweather-radar-addon/issues)
- [Home Assistant Community Forum](https://community.home-assistant.io/)
- [Documentation](https://github.com/yourusername/willyweather-radar-addon)
