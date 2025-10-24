#!/usr/bin/with-contenv bashio

set -e

bashio::log.info "Starting WillyWeather Radar Addon..."

# Check if API key is configured
if ! bashio::config.has_value 'api_key'; then
    bashio::log.fatal "API key not configured!"
    bashio::log.fatal "Please add your WillyWeather API key in the addon configuration"
    bashio::exit.nok
fi

API_KEY=$(bashio::config 'api_key')
CACHE_DURATION=$(bashio::config 'cache_duration')
LOG_LEVEL=$(bashio::config 'log_level')

bashio::log.info "Configuration loaded:"
bashio::log.info "  Cache duration: ${CACHE_DURATION}s"
bashio::log.info "  Log level: ${LOG_LEVEL}"

# Export environment variables
export API_KEY="${API_KEY}"
export CACHE_DURATION="${CACHE_DURATION}"
export LOG_LEVEL="${LOG_LEVEL}"

# Run the Python server
cd /app
bashio::log.info "Starting Python server..."
exec python3 /app/server.py
