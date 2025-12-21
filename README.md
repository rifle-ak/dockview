# DockView - Beautiful Docker Container Dashboard

A gorgeous, modern web dashboard for managing and monitoring your Docker containers. Built specifically for Saltbox and Plex servers, but works with any Docker environment.

![DockView Dashboard](https://img.shields.io/badge/Docker-Container%20Management-2496ED?style=for-the-badge&logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)

## ✨ Features

### 🎨 Beautiful Modern UI
- **Glassmorphism design** with gradient effects and smooth animations
- **Dark theme** optimized for extended viewing
- **Responsive layout** - works perfectly on desktop, tablet, and mobile
- **Smooth hover effects** and transitions throughout
- **Custom scrollbars** with gradient styling

### 📊 Real-Time Monitoring
- **Live container stats** - CPU and memory usage with visual progress bars
- **Color-coded indicators** - green (healthy), yellow (warning), red (critical)
- **Auto-refresh** every 10 seconds
- **Uptime tracking** in human-readable format (days, hours, minutes)
- **Port mappings** display for easy reference

### 🔍 Smart Filtering & Search
- **Real-time search** across container names, images, and status
- **Status filters** - view all, running, or stopped containers
- **Instant results** with smooth transitions

### 🎯 Container Management
- **Start/Stop/Restart** containers with one click
- **Toast notifications** for action feedback
- **Smart action buttons** - shows relevant actions based on container state
- **Safe operations** with error handling

### 🎭 App Recognition
Automatically detects and displays custom icons for 40+ popular applications:
- **Media Servers**: Plex, Jellyfin, Emby, Tdarr
- **Media Management**: Sonarr, Radarr, Lidarr, Prowlarr, Overseerr
- **Download Clients**: Transmission, qBittorrent, SABnzbd, NZBGet
- **Reverse Proxies**: Traefik, Nginx
- **Databases**: PostgreSQL, MySQL, MariaDB, MongoDB, Redis
- **Dashboards**: Portainer, Heimdall, Organizr, Homepage
- **Security**: Pi-hole, AdGuard, Vaultwarden, WireGuard
- **Cloud Storage**: Nextcloud, Photoprism, Immich
- **Monitoring**: Grafana, Prometheus, Netdata
- And many more!

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Access to Docker socket (`/var/run/docker.sock`)
- Traefik reverse proxy (optional, for HTTPS)

### Installation

1. **Clone or download this repository**
   ```bash
   cd /opt
   git clone <your-repo-url> dockview
   cd dockview
   ```

2. **Update docker-compose.yml** (if needed)
   - Change the Traefik host rule to your domain
   - Adjust timezone if needed
   ```yaml
   - "traefik.http.routers.dockview.rule=Host(`dockview.yourdomain.com`)"
   environment:
     - TZ=America/New_York  # Change to your timezone
   ```

3. **Build and run**
   ```bash
   docker-compose up -d --build
   ```

4. **Access your dashboard**
   - With Traefik: https://dockview.yourdomain.com
   - Direct access: http://localhost:5080

### Manual Docker Run (without docker-compose)

```bash
docker build -t dockview .
docker run -d \
  --name dockview \
  -p 5080:5080 \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -e TZ=America/New_York \
  --restart unless-stopped \
  dockview
```

## 📸 Screenshots

### Main Dashboard
- Clean, modern interface with gradient cards
- Real-time CPU and memory monitoring
- Status badges with pulse animations for running containers

### Container Cards
Each card displays:
- Application icon (auto-detected)
- Container name and image
- Status badge with animation
- Uptime information
- CPU usage with color-coded progress bar
- Memory usage with detailed stats
- Exposed ports
- Action buttons (Start/Stop/Restart)

### Features in Action
- **Search**: Instantly filter containers as you type
- **Tabs**: Switch between All/Running/Stopped views
- **Stats**: Live running/stopped container counts in header
- **Notifications**: Toast messages for all actions

## 🛠️ Technical Details

### Backend (Python/FastAPI)
- **FastAPI** framework for high-performance API
- **Docker SDK** for Python to interact with containers
- **Accurate CPU/Memory calculations** using Docker stats API
- **Smart app detection** from container names and images
- **Error handling** and graceful degradation

### Frontend (Vanilla JS)
- **No framework dependencies** - pure JavaScript
- **TailwindCSS** via CDN for styling
- **Font Awesome** icons
- **Responsive grid layout**
- **Smooth animations** with CSS transitions

### Docker Configuration
- **Read-only socket mount** for security
- **Minimal Python image** (python:3.11-slim)
- **Auto-restart** on failures
- **Traefik integration** ready

## 🔧 Configuration

### Environment Variables
- `TZ` - Timezone (default: America/New_York)

### Network Requirements
- Must be on the same network as Traefik (if using)
- Requires access to Docker socket

### Customization

#### Change Refresh Interval
Edit `index.html` line 688:
```javascript
setInterval(fetchContainers, 10000); // 10000ms = 10 seconds
```

#### Add Custom App Icons
Edit `main.py` function `detect_app_type()` to add your app patterns:
```python
app_patterns = {
    'your-app': 'your-app',
    # ... more patterns
}
```

Then edit `index.html` APP_ICONS object to add icon and color:
```javascript
const APP_ICONS = {
    'your-app': { icon: 'fas fa-custom-icon', color: '#hexcolor' },
    // ... more icons
}
```

## 🆚 Improvements Over Original

This version fixes and improves ChatGPT's original implementation:

### Backend Fixes
✅ Now serves the HTML file (was only API endpoints)
✅ Fixed CPU calculation (was showing GB, now shows %)
✅ Fixed memory calculation (was MB value, now shows %)
✅ Added missing restart/stop/start endpoints
✅ Added app type detection
✅ Human-readable uptime formatting
✅ Proper error handling
✅ Port information extraction

### Frontend Improvements
✅ Complete UI redesign with modern aesthetics
✅ Gradient backgrounds and glassmorphism
✅ Smooth animations and transitions
✅ Search functionality
✅ Toast notifications
✅ Loading states
✅ Empty state handling
✅ Pulse animations for running containers
✅ Color-coded progress bars
✅ Stats summary in header
✅ Better responsive design
✅ 40+ app-specific icons

### Docker/Deployment Fixes
✅ Updated Dockerfile to copy all necessary files
✅ Removed non-existent templates volume
✅ Proper requirements.txt usage
✅ Optimized build process

## 🐛 Troubleshooting

### Containers not showing up
- Check Docker socket is mounted: `docker logs dockview`
- Verify permissions on `/var/run/docker.sock`

### Can't access dashboard
- Check container is running: `docker ps | grep dockview`
- Verify port 5080 is exposed: `docker port dockview`
- Check Traefik labels if using reverse proxy

### Stats not updating
- Check browser console for errors (F12)
- Verify container has access to Docker API
- Check auto-refresh is working (10 second interval)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for Saltbox and Plex server environments
- Uses Docker Python SDK
- Styled with TailwindCSS
- Icons from Font Awesome

## 🚦 Status

✅ Fully functional
✅ Production ready
✅ Actively maintained

---

**Enjoy your beautiful Docker dashboard!** 🐳✨
