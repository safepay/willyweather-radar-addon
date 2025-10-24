ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base:3.19
FROM ${BUILD_FROM}

# Install requirements - use system packages where possible
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-pillow \
    py3-numpy \
    py3-requests \
    py3-flask

# Copy application files
COPY rootfs /

# Install remaining Python dependencies with break-system-packages flag
# Note: Only install packages not available via apk
RUN pip3 install --no-cache-dir --break-system-packages \
    flask==3.0.0

# Set working directory
WORKDIR /app

# Make run script executable
RUN chmod a+x /app/run.sh

CMD ["/app/run.sh"]
