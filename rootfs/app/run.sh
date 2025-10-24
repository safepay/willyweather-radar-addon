#!/usr/bin/with-contenv bashio

bashio::log.info "Starting WillyWeather Radar Addon..."

# Check if API key is configured
if ! bashio::config.has_value 'api_key'; then
    bashio::log.fatal "API key not configured!"
    bashio::exit.nok
fi

# Run the Python server
cd /app
exec python3 /app/server.py
