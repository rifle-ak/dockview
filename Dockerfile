FROM python:3.11-slim

WORKDIR /app

# Install system utilities for debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY main.py .
COPY index.html .
COPY requirements.txt .
COPY config.yaml .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Traefik / direct access
EXPOSE 5080

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5080"]