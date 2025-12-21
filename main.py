from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import docker
import re

# Initialize Docker client
client = docker.from_env()

# Initialize FastAPI
app = FastAPI(title="DockView API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def detect_app_type(container_name, image_name):
    """Detect the application type from container name or image"""
    name_lower = container_name.lower()
    image_lower = image_name.lower()

    # Common application mappings
    app_patterns = {
        'plex': 'plex',
        'tdarr': 'tdarr',
        'sonarr': 'sonarr',
        'radarr': 'radarr',
        'lidarr': 'lidarr',
        'prowlarr': 'prowlarr',
        'overseerr': 'overseerr',
        'jellyfin': 'jellyfin',
        'emby': 'emby',
        'traefik': 'traefik',
        'portainer': 'portainer',
        'nginx': 'nginx',
        'postgres': 'postgresql',
        'mysql': 'mysql',
        'mariadb': 'mariadb',
        'redis': 'redis',
        'mongodb': 'mongodb',
        'transmission': 'transmission',
        'qbittorrent': 'qbittorrent',
        'sabnzbd': 'sabnzbd',
        'nzbget': 'nzbget',
        'jackett': 'jackett',
        'homepage': 'homepage',
        'heimdall': 'heimdall',
        'organizr': 'organizr',
        'grafana': 'grafana',
        'prometheus': 'prometheus',
        'netdata': 'netdata',
        'pihole': 'pihole',
        'adguard': 'adguard',
        'nextcloud': 'nextcloud',
        'photoprism': 'photoprism',
        'immich': 'immich',
        'vaultwarden': 'vaultwarden',
        'wireguard': 'wireguard',
        'openvpn': 'openvpn',
    }

    for pattern, app_type in app_patterns.items():
        if pattern in name_lower or pattern in image_lower:
            return app_type

    return 'docker'

def format_uptime(seconds):
    """Format uptime in a human-readable format"""
    if seconds == 0:
        return "Stopped"

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or (days == 0 and hours == 0):
        parts.append(f"{minutes}m")

    return " ".join(parts)

def calculate_cpu_percent(stats):
    """Calculate CPU percentage from Docker stats"""
    try:
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                    stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                       stats['precpu_stats']['system_cpu_usage']
        cpu_count = stats['cpu_stats'].get('online_cpus', 1)

        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
            return round(cpu_percent, 2)
    except (KeyError, ZeroDivisionError):
        pass
    return 0.0

def calculate_memory_percent(stats):
    """Calculate memory percentage from Docker stats"""
    try:
        used_memory = stats['memory_stats']['usage']
        available_memory = stats['memory_stats']['limit']

        if available_memory > 0:
            memory_percent = (used_memory / available_memory) * 100.0
            return round(memory_percent, 2)
    except (KeyError, ZeroDivisionError):
        pass
    return 0.0

def format_bytes(bytes_value):
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def get_containers():
    """Get all containers with their stats"""
    containers_list = []

    for c in client.containers.list(all=True):
        try:
            # Get container stats if running
            stats = c.stats(stream=False) if c.status == "running" else None

            # Calculate uptime
            if c.status == "running" and 'StartedAt' in c.attrs['State']:
                started_at = c.attrs['State']['StartedAt']
                # Parse the timestamp
                if started_at.endswith('Z'):
                    started_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                else:
                    started_time = datetime.fromisoformat(started_at[:26] + '+00:00')
                uptime_seconds = int((datetime.now(timezone.utc) - started_time).total_seconds())
                uptime = format_uptime(uptime_seconds)
            else:
                uptime_seconds = 0
                uptime = "Stopped"

            # Calculate CPU and memory
            if stats:
                cpu = calculate_cpu_percent(stats)
                memory = calculate_memory_percent(stats)
                memory_usage = stats['memory_stats'].get('usage', 0)
                memory_limit = stats['memory_stats'].get('limit', 0)
                memory_text = f"{format_bytes(memory_usage)} / {format_bytes(memory_limit)}"
            else:
                cpu = 0.0
                memory = 0.0
                memory_text = "N/A"

            # Get image name
            image = c.image.tags[0] if c.image.tags else c.attrs['Config']['Image']

            # Detect app type
            app_type = detect_app_type(c.name, image)

            # Get ports
            ports = []
            if c.attrs.get('NetworkSettings', {}).get('Ports'):
                for port, bindings in c.attrs['NetworkSettings']['Ports'].items():
                    if bindings:
                        for binding in bindings:
                            ports.append(f"{binding['HostPort']}:{port}")

            containers_list.append({
                "id": c.id[:12],
                "name": c.name,
                "status": c.status,
                "uptime": uptime,
                "uptime_seconds": uptime_seconds,
                "cpu": cpu,
                "memory": memory,
                "memory_text": memory_text,
                "appType": app_type,
                "image": image,
                "ports": ports
            })
        except Exception as e:
            print(f"Error processing container {c.name}: {e}")
            continue

    # Sort by status (running first) then by name
    containers_list.sort(key=lambda x: (x['status'] != 'running', x['name']))
    return containers_list

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse('index.html')

@app.get("/containers")
async def containers():
    """Get all containers"""
    return get_containers()

@app.post("/restart/{container_id}")
async def restart_container(container_id: str):
    """Restart a container"""
    try:
        container = client.containers.get(container_id)
        container.restart()
        return {"status": "success", "message": f"Container {container_id} restarted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/stop/{container_id}")
async def stop_container(container_id: str):
    """Stop a container"""
    try:
        container = client.containers.get(container_id)
        container.stop()
        return {"status": "success", "message": f"Container {container_id} stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/start/{container_id}")
async def start_container(container_id: str):
    """Start a container"""
    try:
        container = client.containers.get(container_id)
        container.start()
        return {"status": "success", "message": f"Container {container_id} started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5080)
