# Mevin

**Real-time AI video analysis** running locally on your machine. No cloud, no subscriptions, no data leaves your network.

Mevin connects to your cameras (USB, IP/RTSP, video files, phone), feeds frames to a local vision model via [Ollama](https://ollama.com), and streams AI scene descriptions to a live dashboard. Alerts trigger on configurable keywords and notify via Telegram.

\---

## Features

* **Multi-camera support** - USB webcams, RTSP/IP cameras, video files, phone cameras
* **Real-time AI analysis** - streams scene descriptions as they generate
* **Smart camera rotation** - least-recently-analyzed first, focus mode for single camera
* **Motion detection** - skip unchanged frames, save GPU cycles
* **Alert system** - 36 configurable keywords trigger visual + audio + Telegram alerts
* **Horizontal timeline** - video-editor style event view, 3h/6h/12h/24h ranges
* **GPU monitoring** - VRAM, temperature, utilization in the dashboard header
* **Load warnings** - toast notifications when GPU is overloaded
* **Network scanner** - auto-discover USB cameras and RTSP devices on your LAN
* **Video file support** - load any MP4/AVI/MKV as a camera, plays at native FPS with loop
* **Snapshot gallery** - auto-save on alerts, manual capture, browseable gallery
* **Data retention** - auto-cleanup by age, count cap, snapshot size cap, thumbnail stripping
* **Feed persistence** - observations survive page refresh, loaded from database
* **Color-coded feed** - people (blue), danger (red), objects (orange), actions (cyan), safe (green)
* **Setup wizard** - first-launch guide walks through Ollama install and camera setup
* **Auth token** - optional bearer token for LAN access protection
* **FastAPI backend** - production server with auto-generated API docs at `/docs`

## Screenshots

|Dashboard|Timeline|Setup Wizard|
|-|-|-|
|Live camera grid with AI feed|Horizontal event timeline|First-launch configuration|

## Quick Start

### 1\. Install Ollama

Download from [ollama.com](https://ollama.com) and pull a vision model:

```bash
ollama pull gemma3:4b
```

### 2\. Install Mevin

```bash
git clone https://github.com/mephisto1122/mevin.git
cd mevin
pip install -r requirements.txt
```

### 3\. Run

```bash
python mevin.py
```

Open [http://localhost:5555](http://localhost:5555) in your browser. API docs at [http://localhost:5555/docs](http://localhost:5555/docs).

## Requirements

* Python 3.10+
* [Ollama](https://ollama.com) running locally
* A vision-capable model (gemma3:4b recommended)
* A camera source (webcam, IP camera, or video file)
* NVIDIA GPU recommended (works on CPU but slower)

## Supported Models

|Model|Speed|VRAM|Notes|
|-|-|-|-|
|`gemma3:4b`|\~6s/frame|\~4 GB|**Recommended.** Fast, reliable, no thinking mode|
|`gemma4:e2b`|\~3s/frame|\~3 GB|Fastest Gemma 4, has thinking mode (handled)|
|`gemma4:e4b`|\~6s/frame|\~5 GB|Better quality, thinking mode|
|`gemma4:26b`|\~15s/frame|\~15 GB|Near-frontier quality, tight on 16GB|
|`moondream`|\~2s/frame|\~2 GB|Very fast, basic descriptions|
|`qwen2.5vl:3b`|\~4s/frame|\~3 GB|Good alternative|

Mevin auto-detects the correct API endpoint (`/api/chat` vs `/api/generate` vs `/v1/chat/completions`) and handles thinking-mode models by capturing thinking tokens silently.

## Camera Sources

|Type|Source Format|Example|
|-|-|-|
|USB webcam|`0`, `1`, `2`|`0` (first webcam)|
|RTSP camera|`rtsp://user:pass@ip:port/path`|`rtsp://admin:admin@192.168.1.100:554/stream1`|
|HTTP camera|`http://ip:port/video`|`http://192.168.1.50:8080/video`|
|Video file|Full file path|`C:\\Videos\\footage.mp4`|
|Phone (IP Webcam)|`http://phone\_ip:8080/video`|`http://192.168.1.42:8080/video`|

The dashboard includes a **Guides** tab with RTSP URLs for Hikvision, Dahua, TP-Link Tapo, Reolink, Xiaomi, and ONVIF cameras.

## Configuration

All settings are configurable from the dashboard Settings tab. Defaults:

|Setting|Default|Description|
|-|-|-|
|Model|`gemma3:4b`|Ollama vision model|
|Analysis interval|`5s`|Seconds between analyses|
|Max tokens|`200`|Token budget per analysis|
|Image size|`448px`|Inference resolution|
|Motion sensitivity|`0.3%`|Minimum change to trigger analysis|
|Retain days|`30`|Auto-delete observations older than this|
|Max observations|`5000`|Hard cap on stored observations|
|Max snapshot storage|`2000 MB`|Snapshot folder size limit|
|Thumbnail strip|`7 days`|Remove thumbnails from old observations|

### Environment Variables

|Variable|Description|
|-|-|
|`MEVIN\_TOKEN`|Set to protect API routes with bearer token auth|

```bash
# Enable auth
MEVIN\_TOKEN=mysecret python mevin.py

# API calls require token
curl -H "Authorization: Bearer mysecret" http://localhost:5555/api/cameras
# Or query param
curl http://localhost:5555/api/cameras?token=mysecret
```

## Alert Keywords

Default triggers (editable in Settings):

> weapon, knife, gun, fight, fighting, attack, punch, kick, aggressive, threatening, suspicious, intruder, stranger, trespassing, break-in, forced, smash, fallen, falling, unconscious, injured, bleeding, fire, smoke, flame, running, shouting, screaming, panic, unattended, abandoned, mask, covered face, hoodie, loitering, hiding, crawling, climbing, vandalism, theft, stealing, robbery

When a keyword appears in the AI description:

* Feed item gets a red **ALERT** badge
* Alert keywords are shown as red pills
* Audio beep plays in the browser
* Telegram notification sent (if configured)
* Snapshot auto-saved

## Telegram Alerts

1. Message [@BotFather](https://t.me/BotFather) on Telegram, create a bot, get the token
2. Message [@userinfobot](https://t.me/userinfobot) to get your chat ID
3. Enter both in Settings > Telegram Alerts
4. Click "Send Test" to verify

Supports quiet hours (e.g., 23:00-07:00) and minimum interval between alerts (default 30s).

## Architecture

```
mevin.py           FastAPI backend (uvicorn)
dashboard.html     Single-file frontend (vanilla JS, no build step)
monitor.db         SQLite database (auto-created)
snapshots/         Saved camera captures (auto-created)
```

### Backend Components

* **CameraFeed** - OpenCV capture thread per camera, MJPEG streaming
* **VLMAnalyzer** - Ollama API calls, SSE broadcast, smart rotation
* **GPUMonitor** - nvidia-smi polling every 3 seconds
* **DataMaintenance** - periodic cleanup thread

### API Endpoints

All documented at `/docs` (FastAPI auto-generated Swagger UI).

|Method|Endpoint|Description|
|-|-|-|
|GET|`/`|Dashboard|
|GET|`/video/{cam\_id}`|MJPEG stream (resized to 640px)|
|GET|`/events`|SSE event stream|
|GET|`/snapshot/{path}`|Snapshot file|
|GET|`/api/cameras`|List cameras|
|POST|`/api/cameras`|Add camera|
|PUT|`/api/cameras/{id}`|Update camera|
|DELETE|`/api/cameras/{id}`|Remove camera|
|GET|`/api/settings`|Get all settings|
|POST|`/api/settings`|Update settings|
|GET|`/api/focus`|Get focus state|
|POST|`/api/focus`|Set focus camera|
|GET|`/api/gpu`|GPU stats|
|GET|`/api/timeline`|Historical events|
|GET|`/api/recent-feed`|Cached feed items|
|GET|`/api/gallery`|Snapshot list|
|POST|`/api/take-snapshot/{id}`|Manual snapshot|
|POST|`/api/scan`|Start network scan|
|GET|`/api/scan`|Scan status|
|GET|`/api/video-files`|Detected video files|
|GET|`/api/data-stats`|Database stats|
|POST|`/api/cleanup`|Run cleanup now|
|POST|`/api/clear`|Delete all observations|
|POST|`/api/pin/{id}`|Pin observation|
|POST|`/api/unpin/{id}`|Unpin observation|
|POST|`/api/test-telegram`|Test Telegram|

## Data Management

Mevin auto-manages storage:

* **Old observations** deleted after 30 days (configurable)
* **Observation count** capped at 5000 (oldest unpinned removed first)
* **Thumbnails stripped** from observations older than 7 days (saves \~80% DB size)
* **Snapshots** deleted after 30 days or when folder exceeds 2GB
* **Pinned items** are never auto-deleted
* **VACUUM** runs after each cleanup cycle

## Network Deployment

For LAN access behind nginx:

```nginx
server {
    listen 443 ssl;
    server\_name cameras.local;

    location / {
        proxy\_pass http://127.0.0.1:5555;
        proxy\_set\_header Host $host;
        proxy\_set\_header X-Real-IP $remote\_addr;
        proxy\_buffering off;           # Required for SSE
        proxy\_cache off;               # Required for MJPEG
        proxy\_read\_timeout 86400;      # Keep streams alive
    }
}
```

Set `MEVIN\_TOKEN` when exposing on a network.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

* [Ollama](https://ollama.com) - local model runtime
* [FastAPI](https://fastapi.tiangolo.com) - web framework
* [OpenCV](https://opencv.org) - camera capture
* [Google Gemma](https://ai.google.dev/gemma) - vision models

