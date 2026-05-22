<div align="center">

# Mevin

**Real-time AI video analysis — private, local, open-source.**

Point any camera at a scene and get continuous natural-language descriptions of what's happening, with instant alerts when things go wrong.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

No cloud &nbsp;·&nbsp; No subscriptions &nbsp;·&nbsp; Runs on your hardware

</div>

---

## Overview

Mevin connects to your cameras (USB, RTSP, IP, or video files), captures frames, and sends them to a local vision-language model through Ollama. The model describes what it sees in real time. When it detects keywords you care about — weapons, fire, intruders — it triggers alerts and optionally notifies you on Telegram.

Everything stays on your machine. The only network calls are to your own Ollama instance and (optionally) the Telegram API for push notifications.



<img width="2543" height="1353" alt="image" src="https://github.com/user-attachments/assets/bdeec7c6-88db-4151-9eea-83200c2f2763" />
<img width="2559" height="1358" alt="image" src="https://github.com/user-attachments/assets/8ae147fe-235c-48dc-b81f-f466475daec0" />


## Features

**Analysis** — multi-camera round-robin with per-camera focus lock, streaming token-by-token output, motion-gated inference to save GPU cycles, configurable prompt presets, smart response cleaning that strips VLM preamble and thinking blocks.

**Dashboard** — split-panel layout with live event feed, resizable panels, semantic text highlighting (people, actions, objects, danger words), click-to-expand event detail overlay, per-camera danger sparklines, zoomable timeline with smart scroll-to-events.

**Alerts** — keyword matching with negation awareness ("no weapon" won't trigger "weapon"), Telegram notifications with photos, quiet hours, configurable cooldown intervals.

**Infrastructure** — SQLite with WAL mode, async observation writer, GPU monitoring (VRAM/temp/utilization), automatic data retention and cleanup, optional auth token, full REST API with OpenAPI docs.

## Quick start

### 1. Install Ollama and pull a vision model

```bash
ollama pull gemma3:4b
```

Other supported models: `llava`, `llava-llama3`, `moondream`, `minicpm-v` — anything on Ollama that accepts images.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python mevin.py
```

```
============================================================
  MEVIN (FastAPI)
============================================================
  Model:     gemma3:4b
  Cameras:   1
    - Camera 1 (0)
  Dashboard: http://localhost:5555
  API Docs:  http://localhost:5555/docs
  Retention: 30d / max 5000 obs / 2000MB snaps
============================================================
```

Open **http://localhost:5555** — the setup wizard walks you through connecting your first camera.

## Camera sources

| Type | Source format | Example |
|------|--------------|---------|
| USB webcam | Device index | `0` |
| Hikvision | RTSP | `rtsp://admin:PASS@IP:554/Streaming/Channels/102` |
| Dahua | RTSP | `rtsp://admin:PASS@IP:554/cam/realmonitor?channel=1&subtype=1` |
| Tapo | RTSP | `rtsp://USER:PASS@IP:554/stream2` |
| Reolink | RTSP | `rtsp://admin:PASS@IP:554/h264Preview_01_sub` |
| Phone (IP Webcam) | HTTP | `http://PHONE_IP:8080/video` |
| Video file | File path | `/path/to/footage.mp4` |

The built-in scanner can auto-discover USB and network cameras from the dashboard.

## Prompt presets

| Preset | Prompt |
|--------|--------|
| Security | People count, actions, danger signs. Flag anything suspicious. One sentence. |
| Detailed | Describe people, objects, vehicles, and actions in detail. Two sentences max. |
| Alerts Only | Only report if you see danger: violence, fire, weapons, break-in, injury. Otherwise reply: Clear. |
| Traffic | Count vehicles and pedestrians. Note accidents, congestion, or jaywalking. One sentence. |
| Retail | Count customers, note crowding levels, detect potential theft or unusual behavior. One sentence. |
| Home | Identify who is visible, note packages, strangers, or open doors/windows. One sentence. |
| Parking | Count parked cars, note available spots, detect break-ins or accidents. One sentence. |
| Pet Watch | Identify any animals visible, their activity and location. Note if pets are in restricted areas. One sentence. |
| Minimal | One sentence max. What changed? |

Custom prompts are fully supported — the presets are starting points.

## Configuration

All settings are stored in SQLite and editable from the dashboard settings page.

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gemma3:4b` | Ollama model name |
| `ollama_url` | `http://localhost:11434` | Ollama API endpoint |
| `analysis_interval` | `3` | Seconds between analysis cycles |
| `inference_size` | `768` | Max image dimension sent to VLM |
| `max_tokens` | `80` | Max response tokens |
| `motion_sensitivity` | `0.3` | Motion threshold (% pixels changed) |
| `motion_enabled` | `true` | Skip analysis when no motion detected |
| `alert_keywords` | *(40+ default words)* | Comma-separated trigger words |
| `telegram_token` | — | Telegram bot token |
| `telegram_chat_id` | — | Telegram chat ID for notifications |
| `telegram_quiet_start` | — | Quiet hours start (e.g. `23:00`) |
| `telegram_quiet_end` | — | Quiet hours end (e.g. `07:00`) |
| `telegram_min_interval` | `30` | Min seconds between Telegram messages |
| `snapshot_quality` | `90` | JPEG quality for saved snapshots |
| `retain_days` | `30` | Delete events older than N days |
| `retain_max_obs` | `5000` | Max stored observations |
| `retain_max_snap_mb` | `2000` | Max snapshot storage in MB |

### Environment variables

| Variable | Description |
|----------|-------------|
| `MEVIN_TOKEN` | Set to require auth token for all API requests |

## API

Interactive docs available at `/docs` when running. Key endpoints:

```
GET   /                        Dashboard
GET   /video/{cam_id}          MJPEG live stream
GET   /events                  SSE event stream

GET   /api/settings            Read all settings
POST  /api/settings            Update settings

GET   /api/cameras             List cameras
POST  /api/cameras             Add camera
PUT   /api/cameras/{id}        Update camera
DELETE /api/cameras/{id}       Remove camera

GET   /api/timeline?hours=N    Event timeline
GET   /api/recent-feed?limit=N Recent feed items
GET   /api/gallery             Snapshot gallery

GET   /api/focus               Current focus camera
POST  /api/focus               Set focus camera
POST  /api/pause               Pause analysis
POST  /api/resume              Resume analysis

GET   /api/gpu                 GPU stats
POST  /api/take-snapshot/{id}  Manual snapshot
POST  /api/scan                Start camera scanner
GET   /api/scan                Scanner status
GET   /api/data-stats          Storage usage
POST  /api/cleanup             Run data cleanup
POST  /api/test-telegram       Send test notification
```

## Architecture

```
mevin.py            Backend — FastAPI, VLM analyzer, camera threads, SSE
dashboard.html      Frontend — single-file SPA (HTML + CSS + JS)
monitor.db          SQLite database (auto-created at runtime)
snapshots/          Saved alert frames (auto-created at runtime)
```

**Backend threads:** one thread per camera (frame capture), one analyzer thread (round-robin VLM inference), one GPU monitor thread, one observation writer thread (async DB writes), one maintenance thread (periodic cleanup).

**Frontend performance:** single MJPEG stream (thumbnails use analysis snapshots instead of live streams), adaptive-framerate canvas rendering for per-camera lifelines, event-delegated timeline with throttled tooltips, parallel API initialization via `Promise.all`.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with a vision model pulled
- A camera source (USB, RTSP, HTTP, or video file)
- NVIDIA GPU recommended — CPU inference works but is significantly slower

## License

[MIT](LICENSE)
