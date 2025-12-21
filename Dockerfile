FROM python:3.11-slim

WORKDIR /app

# Copy application files
COPY main.py .
COPY index.html .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Traefik / direct access
EXPOSE 5080

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5080"]