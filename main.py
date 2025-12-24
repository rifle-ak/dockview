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
import json
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

# Cache for Pi-hole session ID (valid for ~1 hour)
_pihole_session_cache = {
    'sid': None,
    'timestamp': 0,
    'ttl': 3600  # Session valid for 1 hour
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
        timeout = aiohttp.ClientTimeout(total=6)  # 6-second timeout - balanced for reliability and speed
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
        logger.warning(f"Timeout (6s) fetching data from {service_name} at {full_url}")
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

            # Extract network I/O stats
            network_stats = stats.get('networks', {})
            network_rx = 0
            network_tx = 0
            for interface, data in network_stats.items():
                network_rx += data.get('rx_bytes', 0)
                network_tx += data.get('tx_bytes', 0)
        else:
            cpu = 0.0
            memory = 0.0
            memory_text = "N/A"
            network_rx = 0
            network_tx = 0

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

        # Get health check status
        health_status = None
        if c.attrs.get('State', {}).get('Health'):
            health_status = c.attrs['State']['Health'].get('Status', 'unknown')

        # Get resource limits from HostConfig
        host_config = c.attrs.get('HostConfig', {})

        # CPU limits
        cpu_shares = host_config.get('CpuShares', 0)  # CPU shares (relative weight)
        nano_cpus = host_config.get('NanoCpus', 0)  # CPU limit in nanocpus (1e9 = 1 CPU)
        cpu_limit = None
        if nano_cpus > 0:
            cpu_limit = nano_cpus / 1e9  # Convert to number of CPUs
        elif cpu_shares > 0 and cpu_shares != 1024:  # 1024 is default, means no limit
            cpu_limit = cpu_shares / 1024  # Approximate conversion

        # Memory limit
        memory_limit_bytes = host_config.get('Memory', 0)
        memory_limit = None
        if memory_limit_bytes > 0:
            memory_limit = format_bytes(memory_limit_bytes)

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
            "ports": ports,
            "health": health_status,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "network_rx": network_rx,
            "network_tx": network_tx,
            "network_rx_text": format_bytes(network_rx),
            "network_tx_text": format_bytes(network_tx)
        }
    except Exception as e:
        logger.error(f"Error processing container {c.name}: {e}")
        return None

def get_containers():
    """Get all containers with their stats (parallelized for speed, with caching)"""
    # Check cache first
    current_time = time.time()
    if _container_cache['data'] and (current_time - _container_cache['timestamp']) < _container_cache['ttl']:
        logger.info(f"✓ Returning cached container data (age: {int(current_time - _container_cache['timestamp'])}s / {_container_cache['ttl']}s TTL)")
        return _container_cache['data']

    start_time = time.time()
    containers_list = []
    all_containers = client.containers.list(all=True)

    logger.info(f"⏱ Fetching stats for {len(all_containers)} containers...")

    # Process containers in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=20) as executor:  # Increased from 10 to 20 for better parallelization
        future_to_container = {executor.submit(process_container, c): c for c in all_containers}

        for future in as_completed(future_to_container):
            result = future.result()
            if result:
                containers_list.append(result)

    # Sort by status (running first) then by name
    containers_list.sort(key=lambda x: (x['status'] != 'running', x['name']))

    elapsed = time.time() - start_time
    logger.info(f"✓ Successfully processed {len(containers_list)} containers in {elapsed:.2f}s")

    # Update cache
    _container_cache['data'] = containers_list
    _container_cache['timestamp'] = current_time

    return containers_list

@app.get("/")
async def root():
    """Serve the main HTML page"""
    logger.info("Serving index.html from root endpoint")
    return FileResponse('index.html', media_type='text/html')

@app.get("/manifest.json")
async def manifest():
    """Serve PWA manifest"""
    return FileResponse('manifest.json', media_type='application/json')

@app.get("/sw.js")
async def service_worker():
    """Serve service worker"""
    return FileResponse('sw.js', media_type='application/javascript')

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

@app.get("/logs/{container_id}")
async def get_container_logs(container_id: str, tail: int = 100):
    """Get container logs"""
    try:
        container = client.containers.get(container_id)
        logs = container.logs(tail=tail, timestamps=True).decode('utf-8', errors='replace')
        return {
            "container_id": container_id,
            "container_name": container.name,
            "logs": logs,
            "lines": len(logs.split('\n'))
        }
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/details/{container_id}")
async def get_container_details(container_id: str):
    """Get detailed container information"""
    try:
        container = client.containers.get(container_id)
        attrs = container.attrs
        config = attrs.get('Config', {})
        host_config = attrs.get('HostConfig', {})
        network_settings = attrs.get('NetworkSettings', {})

        return {
            "id": container.id,
            "name": container.name,
            "image": config.get('Image', 'N/A'),
            "created": attrs.get('Created', 'N/A'),
            "status": container.status,
            "env": config.get('Env', []),
            "volumes": host_config.get('Binds', []),
            "ports": network_settings.get('Ports', {}),
            "networks": list(network_settings.get('Networks', {}).keys()),
            "restart_policy": host_config.get('RestartPolicy', {}).get('Name', 'no'),
            "labels": config.get('Labels', {}),
            "command": config.get('Cmd', []),
            "entrypoint": config.get('Entrypoint', [])
        }
    except Exception as e:
        logger.error(f"Error fetching details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== NETWORK TOPOLOGY ENDPOINTS =====

@app.get("/network/topology")
async def get_network_topology():
    """Get Docker network topology"""
    try:
        networks = client.networks.list()
        topology = []

        for network in networks:
            net_data = {
                "id": network.id[:12],
                "name": network.name,
                "driver": network.attrs.get('Driver', 'unknown'),
                "scope": network.attrs.get('Scope', 'unknown'),
                "subnet": None,
                "gateway": None,
                "containers": []
            }

            # Get IPAM config
            ipam = network.attrs.get('IPAM', {})
            if ipam and 'Config' in ipam and ipam['Config']:
                config = ipam['Config'][0] if ipam['Config'] else {}
                net_data['subnet'] = config.get('Subnet')
                net_data['gateway'] = config.get('Gateway')

            # Get connected containers
            containers_dict = network.attrs.get('Containers', {})
            for cont_id, cont_info in containers_dict.items():
                net_data['containers'].append({
                    "id": cont_id[:12],
                    "name": cont_info.get('Name', 'unknown'),
                    "ipv4": cont_info.get('IPv4Address', '').split('/')[0] if cont_info.get('IPv4Address') else None
                })

            topology.append(net_data)

        return topology
    except Exception as e:
        logger.error(f"Error fetching network topology: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/network/scan")
async def scan_network():
    """Scan local network for devices using ARP"""
    try:
        import subprocess
        import re

        devices = []

        # Use arp-scan if available, fallback to arp
        try:
            # Try arp-scan first (more comprehensive)
            result = subprocess.run(
                ['arp-scan', '--localnet', '--quiet'],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse arp-scan output
            for line in result.stdout.split('\n'):
                match = re.match(r'(\d+\.\d+\.\d+\.\d+)\s+([\da-f:]+)\s+(.*)', line, re.I)
                if match:
                    devices.append({
                        "ip": match.group(1),
                        "mac": match.group(2),
                        "hostname": match.group(3) or None,
                        "type": "network_device"
                    })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback to arp command
            result = subprocess.run(
                ['arp', '-a'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse arp output (works on Linux and macOS)
            for line in result.stdout.split('\n'):
                # Match pattern: hostname (IP) at MAC [ether] on interface
                match = re.search(r'(?:(\S+)\s+)?\((\d+\.\d+\.\d+\.\d+)\).*?([\da-f:]{11,17})', line, re.I)
                if match:
                    devices.append({
                        "ip": match.group(2),
                        "mac": match.group(3),
                        "hostname": match.group(1) if match.group(1) and match.group(1) != '?' else None,
                        "type": "network_device"
                    })

        # Try to resolve hostnames for IPs without them
        for device in devices:
            if not device.get('hostname'):
                try:
                    import socket
                    hostname = socket.gethostbyaddr(device['ip'])[0]
                    device['hostname'] = hostname
                except:
                    pass

        logger.info(f"✓ Found {len(devices)} network devices")
        return devices

    except Exception as e:
        logger.error(f"Error scanning network: {e}")
        # Return empty list instead of error to avoid breaking the UI
        return []

# ===== VOLUMES ENDPOINTS =====

@app.get("/volumes")
async def get_volumes():
    """Get all Docker volumes with usage information"""
    try:
        volumes = client.volumes.list()
        volume_list = []

        for volume in volumes:
            vol_data = {
                "name": volume.name,
                "driver": volume.attrs.get('Driver', 'unknown'),
                "mountpoint": volume.attrs.get('Mountpoint', 'N/A'),
                "created": volume.attrs.get('CreatedAt', 'N/A'),
                "scope": volume.attrs.get('Scope', 'local'),
                "labels": volume.attrs.get('Labels') or {},
            }

            # Try to get volume size using docker system df -v
            try:
                import subprocess
                result = subprocess.run(
                    ['docker', 'system', 'df', '-v', '--format', '{{.Name}}\t{{.Size}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                for line in result.stdout.split('\n'):
                    if volume.name in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            vol_data['size'] = parts[1].strip()
                            break

                if 'size' not in vol_data:
                    vol_data['size'] = 'Unknown'
            except:
                vol_data['size'] = 'Unknown'

            # Get containers using this volume
            containers_using = []
            try:
                all_containers = client.containers.list(all=True)
                for container in all_containers:
                    mounts = container.attrs.get('Mounts', [])
                    for mount in mounts:
                        if mount.get('Type') == 'volume' and mount.get('Name') == volume.name:
                            containers_using.append({
                                'id': container.id[:12],
                                'name': container.name,
                                'status': container.status
                            })
                            break
            except:
                pass

            vol_data['containers'] = containers_using
            volume_list.append(vol_data)

        logger.info(f"✓ Found {len(volume_list)} Docker volumes")
        return volume_list

    except Exception as e:
        logger.error(f"Error fetching volumes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== IMAGES ENDPOINTS =====

@app.get("/images")
async def get_images():
    """Get all Docker images"""
    try:
        images = client.images.list()
        image_list = []

        for image in images:
            # Get tags
            tags = image.tags if image.tags else ['<none>:<none>']

            # Get size
            size = format_bytes(image.attrs.get('Size', 0))

            # Get created date
            created_str = image.attrs.get('Created', '')
            try:
                created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                created_ago = format_time_ago(created)
            except:
                created_ago = 'Unknown'

            # Find containers using this image
            containers_using = []
            try:
                all_containers = client.containers.list(all=True)
                for container in all_containers:
                    if container.image.id == image.id:
                        containers_using.append({
                            'id': container.id[:12],
                            'name': container.name,
                            'status': container.status
                        })
            except:
                pass

            for tag in tags:
                repo, tag_name = tag.split(':', 1) if ':' in tag else (tag, 'latest')
                image_list.append({
                    'id': image.id[:12] if hasattr(image, 'id') else 'unknown',
                    'repository': repo,
                    'tag': tag_name,
                    'full_tag': tag,
                    'size': size,
                    'created': created_ago,
                    'containers': containers_using
                })

        logger.info(f"✓ Found {len(image_list)} Docker images")
        return image_list

    except Exception as e:
        logger.error(f"Error fetching images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def format_time_ago(dt):
    """Format datetime as 'X days ago' or 'X hours ago'"""
    now = datetime.now(timezone.utc)
    diff = now - dt

    days = diff.days
    hours = diff.seconds // 3600
    minutes = (diff.seconds % 3600) // 60

    if days > 0:
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

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
        ('portainer', get_portainer_stats()),
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
    """Get Plex stats from Tautulli with rich library info"""
    # Fetch multiple endpoints in parallel for comprehensive stats
    activity = await fetch_service_data('tautulli', '/api/v2', {'cmd': 'get_activity'})
    home_stats = await fetch_service_data('tautulli', '/api/v2', {'cmd': 'get_home_stats', 'time_range': '1'})  # Today
    libraries = await fetch_service_data('tautulli', '/api/v2', {'cmd': 'get_libraries'})

    if not activity:
        return None

    # Current streams
    activity_data = activity.get('response', {}).get('data', {})
    stream_count = activity_data.get('stream_count', 0)
    total_bandwidth = activity_data.get('total_bandwidth', 0)

    # Today's stats (plays, duration)
    plays_today = 0
    duration_today = 0
    if home_stats:
        stats_data = home_stats.get('response', {}).get('data', [])
        for stat in stats_data:
            if stat.get('stat_id') == 'top_movies' or stat.get('stat_id') == 'top_tv':
                # Convert to int safely - handle strings, None, empty strings
                try:
                    total_plays = stat.get('total_plays', 0)
                    plays_today += int(total_plays) if total_plays else 0
                except (ValueError, TypeError):
                    pass  # Skip if conversion fails

                try:
                    total_duration = stat.get('total_duration', 0)
                    duration_today += int(total_duration) if total_duration else 0
                except (ValueError, TypeError):
                    pass  # Skip if conversion fails

    # Library counts
    total_movies = 0
    total_shows = 0
    total_episodes = 0
    if libraries:
        libs_data = libraries.get('response', {}).get('data', [])
        for lib in libs_data:
            section_type = lib.get('section_type', '')
            if section_type == 'movie':
                # Safely convert count to int
                try:
                    count = lib.get('count', 0)
                    total_movies += int(count) if count else 0
                except (ValueError, TypeError):
                    pass
            elif section_type == 'show':
                # Safely convert counts to int
                try:
                    count = lib.get('count', 0)
                    total_shows += int(count) if count else 0
                except (ValueError, TypeError):
                    pass
                try:
                    child_count = lib.get('child_count', 0)
                    total_episodes += int(child_count) if child_count else 0
                except (ValueError, TypeError):
                    pass

    return {
        "stream_count": stream_count,
        "total_bandwidth": total_bandwidth,
        "streams": activity_data.get('sessions', [])[:3],
        "plays_today": plays_today,
        "duration_today": round(duration_today / 3600, 1) if duration_today > 0 else 0,  # Convert to hours
        "total_movies": total_movies,
        "total_shows": total_shows,
        "total_episodes": total_episodes,
        "available": True
    }

async def get_pihole_session():
    """Authenticate with Pi-hole v6 and get session ID"""
    current_time = time.time()

    # Return cached session if still valid
    if _pihole_session_cache['sid'] and (current_time - _pihole_session_cache['timestamp']) < _pihole_session_cache['ttl']:
        return _pihole_session_cache['sid']

    service_config = config.get('services', {}).get('pihole', {})
    password = service_config.get('api_key', '')

    if not password:
        logger.info("Pi-hole: No password configured, trying unauthenticated access")
        return None

    # Pi-hole v6 authentication endpoint
    base_url = service_config.get('url', 'http://pihole:80')
    auth_url = f"{base_url}/api/auth"

    try:
        timeout = aiohttp.ClientTimeout(total=6)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # POST to /api/auth with password to get session ID
            async with session.post(auth_url, json={"password": password}) as response:
                if response.status == 200:
                    auth_data = await response.json()
                    sid = auth_data.get('session', {}).get('sid')

                    if sid:
                        # Cache the session ID
                        _pihole_session_cache['sid'] = sid
                        _pihole_session_cache['timestamp'] = current_time
                        logger.info("🛡️ Pi-hole: Successfully authenticated and obtained session ID")
                        return sid
                    else:
                        logger.warning("Pi-hole: Authentication succeeded but no SID in response")
                        return None
                else:
                    logger.warning(f"Pi-hole: Authentication failed with status {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Pi-hole: Authentication error: {e}")
        return None

@app.get("/widgets/pihole")
async def get_pihole_stats():
    """Get Pi-hole DNS blocking stats (v6+ with session-based auth)"""
    service_config = config.get('services', {}).get('pihole', {})
    base_url = service_config.get('url', 'http://pihole:80')

    # Get session ID for authentication
    sid = await get_pihole_session()

    # Try to fetch stats
    stats_url = f"{base_url}/api/stats/summary"

    try:
        timeout = aiohttp.ClientTimeout(total=6)
        headers = {}

        # Add session ID to headers if available
        if sid:
            headers['X-FTL-SID'] = sid

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(stats_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"🛡️ Pi-hole: Successfully fetched stats - Response: {json.dumps(data, indent=2)[:500]}")

                    # Pi-hole v6 API structure: {queries: {total: N, blocked: N, percent_blocked: X}}
                    # Older versions: {dns_queries_today: N, ads_blocked_today: N, ...}

                    queries = data.get('queries', {})

                    # Try v6 format first (nested under 'queries')
                    queries_today = queries.get('total', 0)
                    blocked_today = queries.get('blocked', 0)
                    percent_blocked = queries.get('percent_blocked', 0)
                    unique_domains = queries.get('unique_domains', 0)

                    # Fallback to older API format if v6 fields are empty
                    if queries_today == 0:
                        queries_today = data.get('dns_queries_today', data.get('queries_today', 0))
                        blocked_today = data.get('ads_blocked_today', data.get('blocked_today', 0))
                        percent_blocked = data.get('ads_percentage_today', data.get('percent_blocked', 0))
                        unique_domains = data.get('domains_being_blocked', data.get('domains_blocked', 0))

                    return {
                        "queries_today": queries_today,
                        "blocked_today": blocked_today,
                        "percent_blocked": round(percent_blocked, 1),
                        "domains_blocked": unique_domains,
                        "available": True
                    }
                elif response.status == 401:
                    # Session expired, clear cache and retry once
                    logger.warning("Pi-hole: Session expired (401), clearing cache")
                    _pihole_session_cache['sid'] = None
                    return None
                else:
                    logger.warning(f"Pi-hole: Stats request failed with status {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Pi-hole: Error fetching stats: {e}")
        return None

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
    """Get Sonarr queue, calendar, and library stats"""
    queue = await fetch_service_data('sonarr', '/api/v3/queue')
    calendar = await fetch_service_data('sonarr', '/api/v3/calendar', {
        'start': datetime.now(timezone.utc).isoformat(),
        'end': (datetime.now(timezone.utc).replace(hour=23, minute=59)).isoformat()
    })
    series = await fetch_service_data('sonarr', '/api/v3/series')

    if queue is None:
        return None

    # Calculate library totals
    total_shows = len(series) if series else 0
    total_episodes = 0
    monitored_shows = 0
    if series:
        for show in series:
            total_episodes += show.get('statistics', {}).get('episodeFileCount', 0)
            if show.get('monitored', False):
                monitored_shows += 1

    return {
        "queue_count": len(queue.get('records', [])) if queue else 0,
        "upcoming_today": len(calendar) if calendar else 0,
        "queue_items": queue.get('records', [])[:5] if queue else [],
        "total_shows": total_shows,
        "total_episodes": total_episodes,
        "monitored_shows": monitored_shows,
        "available": True
    }

@app.get("/widgets/radarr")
async def get_radarr_stats():
    """Get Radarr queue, calendar, and library stats"""
    queue = await fetch_service_data('radarr', '/api/v3/queue')
    calendar = await fetch_service_data('radarr', '/api/v3/calendar', {
        'start': datetime.now(timezone.utc).isoformat(),
        'end': (datetime.now(timezone.utc).replace(hour=23, minute=59)).isoformat()
    })
    movies = await fetch_service_data('radarr', '/api/v3/movie')

    if queue is None:
        return None

    # Calculate library totals
    total_movies = len(movies) if movies else 0
    downloaded_movies = 0
    monitored_movies = 0
    if movies:
        for movie in movies:
            if movie.get('hasFile', False):
                downloaded_movies += 1
            if movie.get('monitored', False):
                monitored_movies += 1

    return {
        "queue_count": len(queue.get('records', [])) if queue else 0,
        "upcoming_today": len(calendar) if calendar else 0,
        "queue_items": queue.get('records', [])[:5] if queue else [],
        "total_movies": total_movies,
        "downloaded_movies": downloaded_movies,
        "monitored_movies": monitored_movies,
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

    # Log the actual structure for debugging
    logger.info(f"🔍 Scrutiny API response structure: {json.dumps(data, indent=2)[:2000]}")

    # Also log what we're checking
    logger.info(f"🔍 Scrutiny checks: has 'data': {'data' in data}, data type: {type(data.get('data'))}, has 'summary': {'summary' in data.get('data', {}) if isinstance(data.get('data'), dict) else False}")

    # Try multiple possible response structures
    total_devices = 0
    critical = 0

    # Track disk health details
    devices_list = []
    avg_temp = 0
    total_temp_count = 0

    # Structure 1: {data: {summary: {"WWN1": {...}, "WWN2": {...}}}} - Dictionary of devices by WWN
    if 'data' in data and isinstance(data['data'], dict) and 'summary' in data['data']:
        summary = data['data']['summary']
        if isinstance(summary, dict):
            # Count the number of device entries (each key is a WWN)
            total_devices = len(summary)
            # Count critical devices (device_status > 0 means issues)
            for wwn, device_data in summary.items():
                device_info = device_data.get('device', {})
                smart = device_data.get('smart', {})

                # Check for critical status or device_status field
                if device_data.get('device_status', 0) > 0 or device_info.get('device_status', 0) > 0:
                    critical += 1

                # Extract disk details
                temp = smart.get('temp', 0)
                if temp > 0:
                    avg_temp += temp
                    total_temp_count += 1

                devices_list.append({
                    "name": device_info.get('device_name', 'Unknown'),
                    "model": device_info.get('model_name', 'Unknown'),
                    "temp": temp,
                    "power_on_hours": smart.get('power_on_hours', 0),
                    "status": device_info.get('device_status', 0)
                })

    # Structure 2: {data: {summary: {total_device_count: N}}} - Count fields
    if total_devices == 0 and 'data' in data and isinstance(data['data'], dict):
        summary = data['data'].get('summary', {})
        total_devices = summary.get('total_device_count', 0)
        critical = summary.get('critical_device_count', 0)

    # Structure 3: {summary: {devices: N}} or direct fields
    if total_devices == 0 and 'summary' in data:
        summary = data['summary']
        total_devices = summary.get('devices', summary.get('total_devices', 0))
        critical = summary.get('critical', summary.get('critical_devices', 0))

    # Structure 4: Direct fields in root
    if total_devices == 0:
        total_devices = data.get('total_devices', data.get('devices', 0))
        critical = data.get('critical', data.get('critical_devices', 0))

    # Structure 5: Array of devices (count them)
    if total_devices == 0 and 'data' in data and isinstance(data['data'], list):
        devices = data['data']
        total_devices = len(devices)
        critical = sum(1 for d in devices if d.get('device_status', 0) > 0)

    # Calculate average temperature
    if total_temp_count > 0:
        avg_temp = round(avg_temp / total_temp_count, 1)

    logger.info(f"💽 Scrutiny: Found {total_devices} total devices, {critical} critical, avg temp: {avg_temp}°C")

    return {
        "total_devices": total_devices,
        "critical": critical,
        "avg_temp": avg_temp,
        "devices": devices_list,
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
        "server": result.get('server_name', 'Unknown'),  # Fixed: server_name is a direct field
        "timestamp": result.get('created_at', ''),
        "available": True
    }

@app.post("/speedtest/run")
async def run_speedtest():
    """Trigger a new speedtest"""
    try:
        # Trigger new speedtest via Speedtest-tracker API
        service_config = config.get('services', {}).get('speedtest', {})

        if not service_config.get('enabled', False):
            raise HTTPException(status_code=400, detail="Speedtest service not enabled")

        url = service_config.get('url')
        api_key = service_config.get('api_key', '')

        if not url:
            raise HTTPException(status_code=400, detail="Speedtest URL not configured")

        if not api_key:
            raise HTTPException(status_code=400, detail="Speedtest API token not configured")

        # Correct endpoint: /api/v1/speedtests/run (requires Bearer token + Accept header)
        full_url = f"{url}/api/v1/speedtests/run"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }

        timeout = aiohttp.ClientTimeout(total=60)  # Speedtest can take up to 60 seconds
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(full_url, headers=headers) as response:
                if response.status in [200, 201]:
                    logger.info("✓ Speedtest triggered successfully")
                    return {"status": "success", "message": "Speedtest started"}
                else:
                    error_text = await response.text()
                    logger.error(f"Speedtest trigger failed with status {response.status}: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Failed to trigger speedtest: {error_text}")
    except asyncio.TimeoutError:
        logger.error("Speedtest trigger timed out")
        raise HTTPException(status_code=504, detail="Speedtest request timed out")
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Error triggering speedtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/widgets/uptime_kuma")
async def get_uptime_kuma_stats():
    """Get service uptime stats from Uptime Kuma status page"""
    service_config = config.get('services', {}).get('uptime_kuma', {})
    slug = service_config.get('slug', 'default')  # Default slug is 'default'

    # Uptime Kuma requires a published status page
    # Endpoint: /api/status-page/{slug}
    data = await fetch_service_data('uptime_kuma', f'/api/status-page/{slug}')

    if not data:
        logger.warning(f"Uptime Kuma: No status page found for slug '{slug}'. Please create and publish a status page in Uptime Kuma.")
        return None

    # Parse status page response
    # Structure: {publicGroupList: [...], config: {...}}
    # Note: Public status page API doesn't include real-time status/heartbeat data
    public_groups = data.get('publicGroupList', [])

    total_monitors = 0
    monitor_names = []

    # Count monitors across all groups
    for group in public_groups:
        monitor_list = group.get('monitorList', [])
        for monitor in monitor_list:
            total_monitors += 1
            monitor_names.append(monitor.get('name', 'Unknown'))

    # Get incident data if available
    incident = data.get('incident')
    has_incident = incident is not None

    logger.info(f"💚 Uptime Kuma: Found {total_monitors} monitors: {', '.join(monitor_names)}")

    return {
        "total_monitors": total_monitors,
        "monitor_names": monitor_names,
        "has_incident": has_incident,
        "status_page_title": data.get('config', {}).get('title', 'Status Page'),
        "available": True
    }

@app.get("/widgets/portainer")
async def get_portainer_stats():
    """Get Docker management stats from Portainer"""
    service_config = config.get('services', {}).get('portainer', {})
    api_key = service_config.get('api_key', '')

    if not api_key:
        logger.warning("Portainer: No API key configured")
        return None

    # Portainer API requires authentication
    # Get endpoints (Docker environments)
    endpoints_data = await fetch_service_data('portainer', '/api/endpoints')

    if not endpoints_data:
        return None

    # Get first endpoint (usually local Docker)
    endpoint_id = endpoints_data[0].get('Id', 1) if endpoints_data else 1

    # Get container count
    containers_data = await fetch_service_data('portainer', f'/api/endpoints/{endpoint_id}/docker/containers/json?all=1')

    # Get stack count
    stacks_data = await fetch_service_data('portainer', '/api/stacks')

    # Get volume count
    volumes_data = await fetch_service_data('portainer', f'/api/endpoints/{endpoint_id}/docker/volumes')

    total_containers = len(containers_data) if containers_data else 0
    running_containers = len([c for c in (containers_data or []) if c.get('State') == 'running'])
    total_stacks = len(stacks_data) if stacks_data else 0
    total_volumes = len(volumes_data.get('Volumes', [])) if volumes_data else 0

    logger.info(f"🐳 Portainer: {total_containers} containers ({running_containers} running), {total_stacks} stacks, {total_volumes} volumes")

    return {
        "total_containers": total_containers,
        "running_containers": running_containers,
        "total_stacks": total_stacks,
        "total_volumes": total_volumes,
        "endpoint_name": endpoints_data[0].get('Name', 'Docker') if endpoints_data else 'Docker',
        "available": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5080)
