# Widget Configuration Guide

## 🔧 Quick Fixes for Common Issues

### Pi-hole Widget

**Issue:** Getting 400/401 errors from Pi-hole API

**Solutions (try in order):**

1. **Option 1: No Password (Recommended for Docker internal network)**
   ```yaml
   pihole:
     enabled: true
     url: "http://pihole:80"
     api_key: ""  # Leave empty
   ```

2. **Option 2: Use Web Password Hash**
   - SSH into Pi-hole container: `docker exec -it pihole bash`
   - Get password hash: `cat /etc/pihole/setupVars.conf | grep WEBPASSWORD`
   - Copy the hash value (looks like: `a1b2c3d4e5f6...`)
   - Use in config:
   ```yaml
   pihole:
     enabled: true
     url: "http://pihole:80"
     api_key: "YOUR_HASHED_PASSWORD_HERE"
   ```

3. **Option 3: Disable Authentication in Pi-hole**
   - In Pi-hole web UI: Settings → API → Allow unauthenticated API requests
   - Then use empty `api_key`

---

### Uptime Kuma Widget

**Issue:** `Cannot connect to host uptime:3001`

**Fix:** Update the hostname in config.yaml:

```yaml
uptime_kuma:
  enabled: true
  url: "http://uptime-kuma:3001"  # Note: 'uptime-kuma' not 'uptime'
  api_key: ""
```

**Note:** Uptime Kuma API requires authentication or public status page. Check your Uptime Kuma settings.

---

### Scrutiny Widget

**Issue:** Not showing data even though API is working

**Check:**
1. Open browser console (F12)
2. Look for: `💽 Scrutiny data: {...}`
3. Verify your config:
   ```yaml
   scrutiny:
     enabled: true
     url: "http://scrutiny:8080"
   ```

---

## ✅ Working Widgets Checklist

After updating config, restart DockView:
```bash
docker compose restart dockview
```

Then check browser console for these logs:
- ✅ `📺 Tautulli data: {...}`
- ✅ `🛡️ Pi-hole data: {...}`
- ✅ `🎫 Overseerr data: {...}`
- ✅ `📺 Sonarr data: {...}`
- ✅ `🎬 Radarr data: {...}`
- ✅ `💿 Tdarr data: {...}`
- ✅ `🔍 Prowlarr data: {...}`
- ✅ `💽 Scrutiny data: {...}`
- ✅ `🎯 Speedtest data: {...}`

---

## 🐛 Debugging Tips

**If widgets show zeros:**
- Check console logs to see actual data received
- Verify the service is actually active (e.g., Plex has streams, queue has items)

**If widgets don't appear:**
- Check `📊 Full widget response:` in console
- If it shows `null`, there's an API/auth issue

**If service returns 401/403:**
- Double-check API key in service settings
- Ensure API key is copied correctly (no extra spaces)

---

## 📝 Example Working Config

```yaml
services:
  tautulli:
    enabled: true
    url: "http://tautulli:8181"
    api_key: "abc123..."

  pihole:
    enabled: true
    url: "http://pihole:80"
    api_key: ""  # Empty for local access

  overseerr:
    enabled: true
    url: "http://overseerr:5055"
    api_key: "xyz789..."

  sonarr:
    enabled: true
    url: "http://sonarr:8989"
    api_key: "def456..."

  radarr:
    enabled: true
    url: "http://radarr:7878"
    api_key: "ghi789..."

  prowlarr:
    enabled: true
    url: "http://prowlarr:9696"
    api_key: "jkl012..."

  scrutiny:
    enabled: true
    url: "http://scrutiny:8080"

  speedtest:
    enabled: true
    url: "http://speedtest-tracker:80"

  uptime_kuma:
    enabled: true
    url: "http://uptime-kuma:3001"
```
