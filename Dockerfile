# Use Python 3.13.6 Alpine 3.21 image - no CVEs and most secure
FROM python:3.13.6-alpine3.21

# Set working directory
WORKDIR /app

# Create non-root user for enhanced security (Alpine syntax)
RUN addgroup -g 1000 appuser && \
    adduser -u 1000 -G appuser -s /bin/sh -D appuser

# Update system packages for Alpine
RUN apk update && \
    apk upgrade && \
    rm -rf /var/cache/apk/*

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with no cache to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Set ownership of application files to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Expose port 8080 for Google Cloud deployment
EXPOSE 8080

# Set environment variables for production deployment
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
# Cloud Run will override PORT, but set default for container
ENV PORT=8080

# Optimized gunicorn configuration for client application
CMD ["gunicorn", \
    "--bind", "0.0.0.0:8080", \
    "--workers", "2", \
    "--threads", "8", \
    "--timeout", "90", \
    "--keep-alive", "10", \
    "--max-requests", "1000", \
    "--max-requests-jitter", "50", \
    "--preload", \
    "--worker-connections", "1000", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "app:app"]
