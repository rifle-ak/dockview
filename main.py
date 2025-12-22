from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import docker
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import yaml
import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Docker client
client = docker.from_env()

# In-memory cache for container stats
_container_cache = {
    'data': None,
    'timestamp': 0,
    'ttl': 10  # Cache for 10 seconds
}

_widget_cache = {
    'data': None,
    'timestamp': 0,
    'ttl': 30  # Cache widgets for 30 seconds
}

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config.yaml not found, using default configuration")
        return {'services': {}, 'widgets': {'show_widgets': False}}
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
        return {'services': {}, 'widgets': {'show_widgets': False}}

config = load_config()

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

# ===== SERVICE INTEGRATION HELPERS =====

async def fetch_service_data(service_name: str, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
    """Fetch data from an external service"""
    service_config = config.get('services', {}).get(service_name, {})

    if not service_config.get('enabled', False):
        return None

    url = service_config.get('url')
    api_key = service_config.get('api_key', '')

    if not url:
        return None

    full_url = f"{url}{endpoint}"
    headers = {}

    # Add API key based on service type
    if api_key:
        if service_name in ['tautulli']:
            params = params or {}
            params['apikey'] = api_key
        elif service_name in ['sonarr', 'radarr', 'prowlarr', 'lidarr']:
            headers['X-Api-Key'] = api_key
        elif service_name in ['overseerr']:
            headers['X-Api-Key'] = api_key
        elif service_name in ['gotify']:
            headers['X-Gotify-Key'] = api_key

    try:
        timeout = aiohttp.ClientTimeout(total=3)  # Fast 3-second timeout for speed
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(full_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully fetched data from {service_name}")
                    return data
                else:
                    logger.warning(f"Service {service_name} returned status {response.status} from {full_url}")
                    return None
    except asyncio.TimeoutError:
        logger.warning(f"Timeout (3s) fetching data from {service_name} at {full_url}")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"Network error fetching from {service_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching from {service_name}: {e}")
        return None

def process_container(c):
    """Process a single container and return its info"""
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

        return {
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
        }
    except Exception as e:
        logger.error(f"Error processing container {c.name}: {e}")
        return None

def get_containers():
    """Get all containers with their stats (parallelized for speed, with caching)"""
    # Check cache first
    current_time = time.time()
    if _container_cache['data'] and (current_time - _container_cache['timestamp']) < _container_cache['ttl']:
        logger.info(f"Returning cached container data ({_container_cache['ttl']}s TTL)")
        return _container_cache['data']

    containers_list = []
    all_containers = client.containers.list(all=True)

    logger.info(f"Fetching stats for {len(all_containers)} containers...")

    # Process containers in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_container = {executor.submit(process_container, c): c for c in all_containers}

        for future in as_completed(future_to_container):
            result = future.result()
            if result:
                containers_list.append(result)

    # Sort by status (running first) then by name
    containers_list.sort(key=lambda x: (x['status'] != 'running', x['name']))

    logger.info(f"Successfully processed {len(containers_list)} containers")

    # Update cache
    _container_cache['data'] = containers_list
    _container_cache['timestamp'] = current_time

    return containers_list

@app.get("/")
async def root():
    """Serve the main HTML page"""
    logger.info("Serving index.html from root endpoint")
    return FileResponse('index.html', media_type='text/html')

@app.get("/containers")
async def containers():
    """Get all containers"""
    logger.info("Fetching containers from /containers endpoint")
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

# ===== WIDGET ENDPOINTS =====

@app.get("/widgets/all")
async def get_all_widgets():
    """Get all widget data in a single request (with caching)"""
    widgets_config = config.get('widgets', {})

    if not widgets_config.get('show_widgets', False):
        return {"enabled": False, "widgets": {}}

    # Check cache first
    current_time = time.time()
    if _widget_cache['data'] and (current_time - _widget_cache['timestamp']) < _widget_cache['ttl']:
        logger.info(f"Returning cached widget data ({_widget_cache['ttl']}s TTL)")
        return _widget_cache['data']

    # Fetch all widgets concurrently with asyncio.gather for maximum speed
    tasks = [
        ('tautulli', get_tautulli_stats()),
        ('pihole', get_pihole_stats()),
        ('overseerr', get_overseerr_stats()),
        ('sonarr', get_sonarr_stats()),
        ('radarr', get_radarr_stats()),
        ('tdarr', get_tdarr_stats()),
        ('prowlarr', get_prowlarr_stats()),
        ('scrutiny', get_scrutiny_stats()),
        ('speedtest', get_speedtest_stats()),
        ('uptime_kuma', get_uptime_kuma_stats()),
    ]

    # Run all tasks concurrently with return_exceptions to not block on failures
    task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

    results = {}
    for (name, _), result in zip(tasks, task_results):
        if isinstance(result, Exception):
            logger.error(f"Error fetching {name} widget: {result}")
            results[name] = None
        else:
            results[name] = result

    response = {
        "enabled": True,
        "widgets": results
    }

    # Update cache
    _widget_cache['data'] = response
    _widget_cache['timestamp'] = current_time

    return response

@app.get("/widgets/tautulli")
async def get_tautulli_stats():
    """Get Plex stats from Tautulli"""
    # Get current activity
    activity = await fetch_service_data('tautulli', '/api/v2', {'cmd': 'get_activity'})
    # Get library stats
    libraries = await fetch_service_data('tautulli', '/api/v2', {'cmd': 'get_libraries'})

    if not activity:
        return None

    data = activity.get('response', {}).get('data', {})

    return {
        "stream_count": data.get('stream_count', 0),
        "total_bandwidth": data.get('total_bandwidth', 0),
        "streams": data.get('sessions', [])[:3],  # Top 3 streams
        "available": True
    }

@app.get("/widgets/pihole")
async def get_pihole_stats():
    """Get Pi-hole DNS blocking stats"""
    # Try v6 API first
    data = await fetch_service_data('pihole', '/api/stats/summary')

    # If that fails, try v5 API
    if not data:
        data = await fetch_service_data('pihole', '/admin/api.php', {'summary': ''})

    if not data:
        return None

    # Handle both v5 and v6 response formats
    summary = data.get('summary', data)  # v6 nests under 'summary', v5 doesn't

    return {
        "queries_today": summary.get('dns_queries_today', 0),
        "blocked_today": summary.get('ads_blocked_today', 0),
        "percent_blocked": round(summary.get('ads_percentage_today', 0), 1),
        "domains_blocked": summary.get('domains_being_blocked', 0),
        "available": True
    }

@app.get("/widgets/overseerr")
async def get_overseerr_stats():
    """Get Overseerr request stats"""
    # Get request count
    requests_data = await fetch_service_data('overseerr', '/api/v1/request', {'take': 10, 'skip': 0})

    if not requests_data:
        return None

    results = requests_data.get('results', [])
    pending = [r for r in results if r.get('status') == 1]  # Status 1 = pending

    return {
        "pending_requests": len(pending),
        "total_requests": requests_data.get('pageInfo', {}).get('results', 0),
        "recent_requests": results[:5],
        "available": True
    }

@app.get("/widgets/sonarr")
async def get_sonarr_stats():
    """Get Sonarr queue and calendar"""
    queue = await fetch_service_data('sonarr', '/api/v3/queue')
    calendar = await fetch_service_data('sonarr', '/api/v3/calendar', {
        'start': datetime.now(timezone.utc).isoformat(),
        'end': (datetime.now(timezone.utc).replace(hour=23, minute=59)).isoformat()
    })

    if queue is None:
        return None

    return {
        "queue_count": len(queue.get('records', [])) if queue else 0,
        "upcoming_today": len(calendar) if calendar else 0,
        "queue_items": queue.get('records', [])[:5] if queue else [],
        "available": True
    }

@app.get("/widgets/radarr")
async def get_radarr_stats():
    """Get Radarr queue and calendar"""
    queue = await fetch_service_data('radarr', '/api/v3/queue')
    calendar = await fetch_service_data('radarr', '/api/v3/calendar', {
        'start': datetime.now(timezone.utc).isoformat(),
        'end': (datetime.now(timezone.utc).replace(hour=23, minute=59)).isoformat()
    })

    if queue is None:
        return None

    return {
        "queue_count": len(queue.get('records', [])) if queue else 0,
        "upcoming_today": len(calendar) if calendar else 0,
        "queue_items": queue.get('records', [])[:5] if queue else [],
        "available": True
    }

@app.get("/widgets/tdarr")
async def get_tdarr_stats():
    """Get Tdarr transcoding stats"""
    data = await fetch_service_data('tdarr', '/api/v2/status')

    if not data:
        # Try alternative endpoint
        data = await fetch_service_data('tdarr', '/api/v2/get-stats')

    if not data:
        return None

    # Tdarr API structure varies, adapt as needed
    return {
        "queue_count": data.get('table1Count', 0),
        "processing": data.get('processing', 0),
        "workers": data.get('workers', []),
        "available": True
    }

@app.get("/widgets/prowlarr")
async def get_prowlarr_stats():
    """Get Prowlarr indexer health"""
    indexers = await fetch_service_data('prowlarr', '/api/v1/indexer')

    if not indexers:
        return None

    enabled = [i for i in indexers if i.get('enable', False)]
    healthy = [i for i in enabled if not i.get('tags', [])]

    return {
        "total_indexers": len(indexers),
        "enabled_indexers": len(enabled),
        "healthy_indexers": len(healthy),
        "available": True
    }

@app.get("/widgets/scrutiny")
async def get_scrutiny_stats():
    """Get disk health from Scrutiny"""
    data = await fetch_service_data('scrutiny', '/api/summary')

    if not data:
        return None

    return {
        "total_devices": data.get('data', {}).get('summary', {}).get('total_device_count', 0),
        "critical": data.get('data', {}).get('summary', {}).get('critical_device_count', 0),
        "available": True
    }

@app.get("/widgets/speedtest")
async def get_speedtest_stats():
    """Get network speed from Speedtest-tracker"""
    # Get latest speedtest result
    data = await fetch_service_data('speedtest', '/api/speedtest/latest')

    if not data:
        return None

    # Speedtest-tracker API structure
    result = data.get('data', {})

    return {
        "download": round(result.get('download', 0), 2),  # Mbps
        "upload": round(result.get('upload', 0), 2),  # Mbps
        "ping": round(result.get('ping', 0), 2),  # ms
        "server": result.get('server', {}).get('name', 'Unknown'),
        "timestamp": result.get('created_at', ''),
        "available": True
    }

@app.get("/widgets/uptime_kuma")
async def get_uptime_kuma_stats():
    """Get service uptime stats from Uptime Kuma"""
    # Get monitors status
    data = await fetch_service_data('uptime_kuma', '/api/status-page/heartbeat')

    if not data:
        # Try alternative endpoint
        data = await fetch_service_data('uptime_kuma', '/metrics')

    if not data:
        return None

    # Uptime Kuma API structure varies, return basic info
    return {
        "monitors": data.get('monitorList', []),
        "up_count": len([m for m in data.get('monitorList', []) if m.get('active')]) if 'monitorList' in data else 0,
        "total_count": len(data.get('monitorList', [])) if 'monitorList' in data else 0,
        "available": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5080)
