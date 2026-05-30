<div align="center">

# Mevin

**Real-time situational video understanding — private, local, open-source.**

Mevin doesn't just describe what's on camera. It reads the *situation* — who's there, what they're doing, what their intent looks like, and whether things are escalating or calming — in real time, on your own hardware.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/license-Personal%20Use-yellow?style=flat-square)](LICENSE)

No cloud &nbsp;·&nbsp; No subscriptions &nbsp;·&nbsp; Runs on your hardware

</div>

---

## What makes it different

Most camera software detects *objects* — "person, car, bag." Mevin reads *situations*. It connects each camera to a local vision-language model through [Ollama](https://ollama.com), but wraps it in a situational brain:

- **It learns what's normal** for each camera, then reasons about what *changed*.
- **It remembers** the last several moments, so it understands a sequence — "a person entered, then started forcing the door" — not just a single frame.
- **It tracks a trajectory** — whether danger is *escalating*, *stable*, or *calming*.
- **It connects cameras** watching the same area into one big-picture narrative.

Everything stays on your machine. The only network calls are to your own Ollama instance and, optionally, a push service for phone alerts.

<img width="2543" height="1353" alt="Board view" src="https://github.com/user-attachments/assets/bdeec7c6-88db-4151-9eea-83200c2f2763" />
<img width="452" height="240" alt="0530 (1)(1)" src="https://github.com/user-attachments/assets/0c9dcfca-33e4-4a34-9cd7-3d7be58ef624" />




<img width="1759" height="1307" alt="Settings" src="https://github.com/user-attachments/assets/25ce3220-0d86-4c13-a7e8-327082d90cbd" />

## Quick start

**One command** — the installer checks Python + Ollama, installs everything, pulls the model, and launches:

```bash
# Mac / Linux
./install.sh

# Windows — double-click install.bat
```

**Manual:**

```bash
ollama pull gemma3:4b
pip install -r requirements.txt
python mevin.py
```

Then open **http://localhost:5555**. A setup wizard walks you through your first camera.

```
============================================================
  MEVIN - real-time situational video analysis
============================================================
  Model:       gemma3:4b  (50 tokens)
  Mode:        Situational (reads intent + trend)
  Analysis:    parallel per-camera, 2s interval
  Cameras:     1
    - Camera 1 (0)
  Workers:     separate process (PID 12345) - GIL isolated
  Dashboard:   http://localhost:5555
  API Docs:    http://localhost:5555/docs
============================================================
```

> **Models:** `gemma3:4b` is the default — solid, reliable descriptions. For maximum speed on many cameras switch to `moondream` (~1–2 s/frame); it auto-gets a compact prompt. Anything on Ollama that accepts images works (`llava`, `minicpm-v`, etc.).

## Connecting cameras

**Auto-discover (easiest)** — click **Discover** in the app. Mevin finds Hikvision, Dahua, Reolink, NVRs, and any ONVIF camera on your network, pulls their stream URLs automatically, and adds every channel in one click — no URLs to type. Enable it once:

```bash
pip install -r requirements-onvif.txt
```

**Manual** — click **Add** and enter a webcam index (`0`), an RTSP URL, or a video-file path:

| Type | Source format | Example |
|------|--------------|---------|
| USB webcam | Device index | `0` |
| Hikvision / NVR | RTSP | `rtsp://admin:PASS@IP:554/Streaming/Channels/102` |
| Dahua | RTSP | `rtsp://admin:PASS@IP:554/cam/realmonitor?channel=1&subtype=1` |
| Reolink | RTSP | `rtsp://admin:PASS@IP:554/h264Preview_01_sub` |
| Tapo | RTSP | `rtsp://USER:PASS@IP:554/stream2` |
| Phone (IP Webcam) | HTTP | `http://PHONE_IP:8080/video` |
| Video file | File path | `/path/to/footage.mp4` |

## Phone access

- **Same WiFi:** open `http://<PC-IP>:5555` on your phone — the dashboard is mobile-responsive.
- **Anywhere:** install [Tailscale](https://tailscale.com) (free) on the PC and phone — reach Mevin remotely with no port-forwarding.
- **Push alerts:** in Settings, set an **ntfy** topic (install the [ntfy app](https://ntfy.sh), subscribe, done — no account) or add a **Telegram** bot token. Alerts fan out to both.

## How it works

### Situational analysis

```
Camera frame
   │  motion check (skips static frames before they reach the GPU)
   ▼
VLM (moondream / gemma3) ── told the camera's baseline + recent history
   │
   ▼
Scene Memory ── learns "normal", tracks danger trend (rising/stable/falling)
   │
   ▼
Board + alerts + phone push
```

Each camera keeps its own **Scene Memory**: an auto-learned baseline of what's normal, a rolling history of recent moments, and a danger trail that yields the *trend*. The model is handed this context, so its answer is situational — "a customer who was browsing has collapsed and isn't moving; escalating" — not a flat caption. The context adapts to the model: small models like moondream get a compact prompt, larger ones get the full reasoning block.

### Scene matching (the big picture)

Assign cameras to a **zone**. When a zone has two or more cameras watching the same area, Mevin periodically combines their individual reads into one overall narrative — "a person left the building (Cam 1), crossed the lot (Cam 2), and is now waiting by the gate (Cam 3)." Cameras stop being separate feeds and become one understanding.

### Real-time, process-isolated

Analysis runs in a **separate process** (`mevin_worker.py`), launched automatically. Python has one lock per process (the GIL); by splitting analysis from the web server, heavy VLM work never freezes the dashboard.

```
┌─ web process (mevin.py) ──────┐        ┌─ worker process ──────────┐
│ cameras, MJPEG, dashboard,    │  HTTP  │ motion + VLM + scene      │
│ SSE, API, SQLite              │◄──────►│ memory + zone synthesis   │
│  • owns the cameras           │ frames │  • own GIL, fully isolated│
│  • relays worker events → SSE │ events │  • fetches frames, never  │
│                               │        │    opens cameras itself   │
└───────────────────────────────┘        └───────────────────────────┘
        shared state via SQLite (WAL) + a heartbeat
```

Three things make it both fast and safe:

- **Frames over localhost** — the worker requests frames from the web process instead of opening cameras itself, so there's no double-open and camera handling stays in one place.
- **Always live** — each analysis fetches the *freshest* frame the instant the GPU is free, so the VLM sees the current moment, never a frame that went stale waiting in the queue.
- **Heartbeat fallback** — if the worker dies, the web process detects the stale heartbeat and resumes analysis in-thread automatically.

Cameras are analyzed **in parallel**, paced by `max(interval, inference time)` so the GPU is never idle when busy. A semaphore serializes inference on a single GPU; raise `MEVIN_GPU_SLOTS` to run several at once on a bigger card. Set `MEVIN_WORKERS=0` to run everything in one process.

### Two views

- **Board** — every camera as a self-contained card: live tile, its own danger lifeline, an always-visible pinned situation with an ESCALATING/CALMING badge, and its own recent feed. Zones show a "Big Picture" banner that turns red when danger is high.
- **Focus** — one big camera with the global feed and a zoomable timeline, for deep-dive.

### Lightweight streaming

Each camera is encoded **once** by a shared background encoder; all viewers read that buffer, so ten browser tabs cost one encode, not ten. Board tiles poll snapshots (looks live, ~20× lighter than holding live MJPEG); only the focused camera uses a true live stream.

## Prompt presets

| Preset | Reads for |
|--------|-----------|
| Situational | Who's present, what they're doing, intent; flags crime live |
| Crime Watch | Concealing/stealing, forcing entry, fighting, casing the area |
| Security | General danger and anything out of place |
| Detailed | Full description — clothing, objects, vehicles |
| Alerts Only | Silent unless there's danger or crime |
| Theft & Retail | Concealment, tag removal, leaving without paying |
| Intrusion | Climbing, prying, sneaking, entering restricted areas |
| Crowd Safety | Density, panic, crush risk |
| Person Safety | Following, cornering, dragging, distress |
| Traffic | Flow, collisions, vehicles where they shouldn't be |
| Home | Deliveries, strangers, open doors/windows |

Custom prompts are fully supported — presets are starting points.

## Configuration

All settings live in SQLite and are editable from the dashboard.

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gemma3:4b` | Ollama vision model |
| `ollama_url` | `http://localhost:11434` | Ollama API endpoint |
| `situational` | `true` | Read situations (baseline + history + trend) |
| `zone_synthesis` | `true` | Combine same-zone cameras into one big picture |
| `analysis_interval` | `2` | Seconds between analyses per camera |
| `inference_size` | `768` | Max image dimension sent to the VLM |
| `max_tokens` | `50` | Max response tokens |
| `motion_sensitivity` | `0.3` | Motion threshold (% pixels changed) |
| `motion_enabled` | `true` | Skip analysis when nothing moves |
| `alert_keywords` | *(76 crime/behavior words)* | Trigger words (negation-aware) |
| `ntfy_topic` | — | ntfy push topic |
| `ntfy_server` | `https://ntfy.sh` | ntfy server |
| `telegram_token` / `telegram_chat_id` | — | Telegram push |
| `telegram_quiet_start` / `_end` | — | Quiet hours |
| `retain_days` | `30` | Delete events older than N days |
| `retain_max_obs` | `5000` | Max stored observations |
| `retain_max_snap_mb` | `2000` | Max snapshot storage (MB) |

### Environment variables

| Variable | Description |
|----------|-------------|
| `MEVIN_WORKERS` | `1` (default) runs the isolated analysis process; `0` analyzes in-process |
| `MEVIN_GPU_SLOTS` | Concurrent VLM inferences (default `1`; raise for multi-GPU / big cards) |
| `MEVIN_TOKEN` | Set to require an auth token on all API requests |

## API

Interactive docs at `/docs`. Selected endpoints:

```
GET   /                          Dashboard
GET   /video/{cam_id}            MJPEG live stream
GET   /snapshot-live/{cam_id}    Single current JPEG (board tiles poll this)
GET   /events                    SSE event stream

GET   /api/settings              Read settings        POST  /api/settings
GET   /api/cameras               List cameras         POST  /api/cameras
PUT   /api/cameras/{id}          Update camera        DELETE /api/cameras/{id}

GET   /api/zones                 Zones and their cameras
GET   /api/situations            Current per-camera situation + zone big-pictures
GET   /api/timeline?hours=N      Event timeline
GET   /api/recent-feed?limit=N&camera_id=  Recent feed (optionally per camera)

GET   /api/onvif-available       Is ONVIF discovery installed
POST  /api/onvif-discover        Start network scan   GET /api/onvif-discover  (status)
POST  /api/onvif-connect         Get a device's streams
POST  /api/onvif-add-all         Add all channels of a device

POST  /api/focus                 Set focus camera     POST /api/pause  /api/resume
GET   /api/gpu                   GPU stats
POST  /api/test-telegram         Test Telegram        POST /api/test-ntfy   Test ntfy
POST  /api/scan                  USB/network scanner   GET /api/scan  (status)
```

## Test footage

The [UCF Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) (1,900 real CCTV clips, 13 categories) is the gold standard for trying Mevin against real scenarios. For quick tests, free clips are on [Pixabay](https://pixabay.com/videos/search/cctv/) and [Mixkit](https://mixkit.co/free-stock-video/cctv/).

To split a long compilation into individual clips by scene:

```bash
python -m yt_dlp -f "worst[ext=mp4]" -o cctv.mp4 "VIDEO_URL"
python tools/split_clips.py cctv.mp4
```

## Project layout

```
mevin.py            Backend — FastAPI, cameras, MJPEG, SSE, API, SQLite, scene memory, zones
mevin_worker.py     Analysis worker — separate process, GIL-isolated VLM pipeline
dashboard.html      Frontend — single-file app (Board + Focus views)
install.sh / .bat   One-command installers
tools/split_clips.py  Scene-split a compilation into individual clips
monitor.db          SQLite (auto-created)       snapshots/  Alert frames (auto-created)
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with a vision model pulled
- A camera source (USB, RTSP, HTTP, or video file)
- NVIDIA GPU recommended — CPU inference works but is much slower
- Optional: `onvif-zeep` + `wsdiscovery` for auto-discovery (`requirements-onvif.txt`)

## License

Personal Use Only. See [LICENSE](LICENSE). For commercial licensing, please contact the author.
