# Use Python 3.13.5 slim image based on Debian Bookworm
FROM python:3.13.5-slim-bookworm

# Set working directory
WORKDIR /app

# Create non-root user for enhanced security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Update system packages and clean up package cache
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

# Start the Flask application
CMD ["python", "app.py"]
