FROM python:3.11-slim

WORKDIR /app

# Copy main.py into the container
COPY main.py .

# Install FastAPI + Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn jinja2 docker

# Expose the port for Traefik / direct access
EXPOSE 5080

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5080"]