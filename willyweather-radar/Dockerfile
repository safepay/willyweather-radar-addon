ARG BUILD_FROM
FROM $BUILD_FROM

# Install requirements
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-pillow \
    py3-numpy \
    py3-requests

# Copy application files
COPY rootfs /

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    flask==3.0.0 \
    requests==2.31.0 \
    pillow==10.1.0 \
    numpy==1.26.2

# Set working directory
WORKDIR /app

# Make run script executable
RUN chmod a+x /app/run.sh

CMD ["/app/run.sh"]
