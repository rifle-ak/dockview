from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import docker

# Initialize Docker client
client = docker.from_env()

# Initialize FastAPI
app = FastAPI(title="Dockview API")

# Allow your frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_containers():
    containers_list = []
    for c in client.containers.list(all=True):
        try:
            stats = c.stats(stream=False) if c.status == "running" else {}
            uptime = int((datetime.utcnow() - datetime.strptime(c.attrs['State']['StartedAt'][:19], "%Y-%m-%dT%H:%M:%S")).total_seconds()) if c.status == "running" else 0
            cpu = round(stats.get('cpu_stats', {}).get('cpu_usage', {}).get('total_usage', 0) / 1_000_000_000, 2)
            memory = round(stats.get('memory_stats', {}).get('usage', 0) / (1024*1024), 2)
        except Exception:
            uptime = 0
            cpu = 0.0
            memory = 0.0

        containers_list.append({
            "id": c.id,
            "name": c.name,
            "status": c.status,
            "uptime": uptime,
            "cpu": cpu,
            "memory": memory
        })
    return containers_list

@app.get("/")
async def root():
    return {"message": "Dockview API is running"}

@app.get("/containers")
async def containers():
    return get_containers()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5080)
