"""
Mevin | Real-time AI video analysis.
pip install fastapi uvicorn opencv-python requests numpy python-multipart
python mevin.py -> http://localhost:5555
API docs -> http://localhost:5555/docs
"""
import asyncio,base64,json,os,pathlib,re,socket,sqlite3,subprocess,threading,time
from collections import deque
from datetime import datetime,timedelta
from queue import Empty,Queue
from typing import Optional
import cv2,numpy as np,requests as req
_NEG_RE=re.compile(r"\b(no|not|don't|didn't|doesn't|without|never|absence|lack|free of|no sign of|isn't|aren't|wasn't|weren't)\b",re.I)

# ===============================================================================
# ONVIF AUTO-DISCOVERY - find cameras & NVRs on the network, no URL typing
# Optional: works if onvif-zeep + wsdiscovery are installed, degrades gracefully.
# ===============================================================================
_ONVIF_OK=False
try:
    from onvif import ONVIFCamera as _ONVIFCamera
    from wsdiscovery.discovery import ThreadedWSDiscovery as _WSDiscovery
    _ONVIF_OK=True
except Exception:
    pass

def onvif_discover(timeout=4):
    """Broadcast WS-Discovery, return ONVIF devices found on the LAN."""
    if not _ONVIF_OK:return []
    found=[]
    try:
        wsd=_WSDiscovery();wsd.start()
        services=wsd.searchServices(timeout=timeout)
        seen=set()
        for svc in services:
            for addr in svc.getXAddrs():
                m=re.search(r"https?://([\d.]+)(?::(\d+))?",addr)
                if not m:continue
                ip=m.group(1);port=int(m.group(2) or 80)
                if ip in seen:continue
                seen.add(ip)
                found.append({"ip":ip,"port":port,"xaddr":addr})
        wsd.stop()
    except Exception as e:print(f"[ONVIF] discovery error: {e}")
    return found

def onvif_streams(ip,port,user,password):
    """Connect to an ONVIF device, return RTSP stream URLs for every channel/profile.
    Returns list of {name, url}. Embeds credentials in the URL so OpenCV can open it."""
    if not _ONVIF_OK:return{"ok":False,"error":"ONVIF not installed","streams":[]}
    try:
        cam=_ONVIFCamera(ip,port,user,password)
        media=cam.create_media_service()
        profiles=media.GetProfiles()
        streams=[];seen=set()
        for i,p in enumerate(profiles):
            try:
                req_=media.create_type("GetStreamUri")
                req_.ProfileToken=p.token
                req_.StreamSetup={"Stream":"RTP-Unicast","Transport":{"Protocol":"RTSP"}}
                uri=media.GetStreamUri(req_).Uri
            except Exception:
                continue
            # Inject credentials into the URL: rtsp://user:pass@ip:554/...
            if user and "@" not in uri:
                uri=re.sub(r"(rtsp://)",f"\\1{user}:{password}@",uri,count=1)
            if uri in seen:continue
            seen.add(uri)
            nm=getattr(p,"Name",None) or f"Channel {i+1}"
            streams.append({"name":str(nm),"url":uri})
        return{"ok":True,"streams":streams}
    except Exception as e:
        return{"ok":False,"error":str(e),"streams":[]}

import uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,JSONResponse,StreamingResponse,Response

PORT=5555;HOST="0.0.0.0";DB_PATH="monitor.db";SNAPSHOT_DIR="snapshots"
STREAM_FPS=15;THUMB_WIDTH=280;JPEG_QUALITY=65


DEFAULTS={
    "model":"gemma3:4b","ollama_url":"http://localhost:11434",
    "prompt":"Describe who is present and exactly what they are doing. Flag any suspicious or criminal behavior: concealing items, forcing entry, fighting, grabbing someone, fleeing, hiding, or casing the area. One sentence.",
    "situational":"true","zone_synthesis":"true",
    "alert_keywords":json.dumps(["weapon","knife","gun","armed","fight","fighting","attack","attacking","punch","punching","kick","kicking","assault","aggressive","threatening","threaten","struggle","struggling","intruder","stranger","trespassing","break-in","breaking","forced","forcing","pry","prying","smash","smashing","climbing","crawling","fallen","falling","collapsed","unconscious","injured","bleeding","fire","smoke","flame","explosion","running","fleeing","chasing","shouting","screaming","panic","crowd","crush","unattended","abandoned","mask","masked","covered face","hoodie","loitering","lurking","hiding","concealing","conceal","stealing","steal","theft","shoplifting","robbery","robbing","grab","grabbing","snatching","vandalism","destroying","cornered","following","stalking","drag","dragging"]),
    "max_tokens":"50","inference_size":"768","analysis_interval":"2",
    "telegram_token":"","telegram_chat_id":"","telegram_on_alert":"true","telegram_on_all":"false",
    "telegram_quiet_start":"","telegram_quiet_end":"","telegram_min_interval":"30",
    "ntfy_topic":"","ntfy_server":"https://ntfy.sh","ntfy_min_interval":"30",
    "motion_sensitivity":"0.3","motion_enabled":"true",
    "snapshot_on_every":"false","snapshot_quality":"90",
    "feed_show_stable":"true","feed_max_items":"200","setup_done":"false",
    "retain_days":"30","retain_max_obs":"5000","retain_max_snap_mb":"2000",
    "cleanup_interval_min":"60","thumb_retain_days":"7",
}

app=FastAPI(title="Mevin",description="Real-time AI video analysis",docs_url="/docs")

# Suppress noisy Windows connection reset errors
import logging
class _ConnFilter(logging.Filter):
    def filter(self,record):
        msg=record.getMessage()
        return "forcibly closed" not in msg and "ConnectionReset" not in msg and "WinError 10054" not in msg
for _ln in ["uvicorn.error","uvicorn","asyncio"]:
    logging.getLogger(_ln).addFilter(_ConnFilter())
os.makedirs(SNAPSHOT_DIR,exist_ok=True)
def sint(v,d=0):
    try:return int(v)
    except:return d
def sfloat(v,d=0.0):
    try:return float(v)
    except:return d

# ===============================================================================
# DATABASE
# ===============================================================================
_db_lock=threading.Lock()
_db_conn=None
def get_shared_db():
    global _db_conn
    if _db_conn is None:
        _db_conn=sqlite3.connect(DB_PATH,check_same_thread=False,timeout=30)
        _db_conn.row_factory=sqlite3.Row
        _db_conn.execute("PRAGMA journal_mode=WAL")
        _db_conn.execute("PRAGMA busy_timeout=5000")
        _db_conn.execute("PRAGMA synchronous=NORMAL")  # faster writes
        _db_conn.execute("PRAGMA cache_size=-4000")     # 4MB cache
    return _db_conn
def db_exec(sql,params=(),fetch=False,fetchone=False):
    acquired=_db_lock.acquire(timeout=5)
    if not acquired:
        print("[DB] Lock timeout, skipping")
        return [] if fetch else None if fetchone else 0
    try:
        c=get_shared_db()
        r=c.execute(sql,params)
        if fetchone:return r.fetchone()
        if fetch:return r.fetchall()
        c.commit();return r.rowcount
    finally:_db_lock.release()
def db_exec_script(sql):
    with _db_lock:
        c=get_shared_db();c.executescript(sql)
def init_db():
    db_exec_script("""
        CREATE TABLE IF NOT EXISTS cameras(id TEXT PRIMARY KEY,name TEXT,source TEXT,enabled INTEGER DEFAULT 1,sort_order INTEGER DEFAULT 0,zone TEXT DEFAULT 'Main');
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,camera_id TEXT,camera_name TEXT,timestamp TEXT,text TEXT,is_alert INTEGER DEFAULT 0,alert_keywords TEXT DEFAULT '[]',snapshot_path TEXT,thumb_b64 TEXT,pinned INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_obs_cam ON observations(camera_id,timestamp DESC);
    """)
    # Migrate: add zone column to existing camera tables
    try:
        cols=[r["name"] for r in db_exec("PRAGMA table_info(cameras)",fetch=True)]
        if "zone" not in cols:
            db_exec("ALTER TABLE cameras ADD COLUMN zone TEXT DEFAULT 'Main'")
            print("[DB] Added zone column to cameras")
    except Exception as e:print(f"[DB] migration: {e}")
    if db_exec("SELECT COUNT(*) FROM cameras",fetchone=True)[0]==0:
        db_exec("INSERT INTO cameras(id,name,source,enabled,sort_order,zone)VALUES('cam0','Camera 1','0',1,0,'Main')")
def get_setting(k):
    r=db_exec("SELECT value FROM settings WHERE key=?",(k,),fetchone=True)
    return r["value"] if r else DEFAULTS.get(k,"")
def set_setting(k,v):
    db_exec("REPLACE INTO settings(key,value)VALUES(?,?)",(k,str(v)))
    _settings_cache.clear()  # invalidate cache

# Settings cache - avoid DB reads on every analysis cycle
_settings_cache={}
_settings_ts=0
def get_all_settings():
    global _settings_ts
    now=time.time()
    if _settings_cache and now-_settings_ts<2:  # cache for 2 seconds
        return dict(_settings_cache)
    rows=db_exec("SELECT key,value FROM settings",fetch=True)
    s=dict(DEFAULTS)
    for r in rows:s[r["key"]]=r["value"]
    _settings_cache.clear();_settings_cache.update(s)
    _settings_ts=now
    return dict(s)
_cameras_cache=[];_cameras_ts=0
def get_cameras_from_db():
    global _cameras_ts
    now=time.time()
    if _cameras_cache and now-_cameras_ts<2:
        return list(_cameras_cache)
    rows=db_exec("SELECT * FROM cameras WHERE enabled=1 ORDER BY sort_order",fetch=True)
    result=[dict(r) for r in rows]
    _cameras_cache.clear();_cameras_cache.extend(result)
    _cameras_ts=now
    return result
def invalidate_cameras():
    _cameras_cache.clear()
    global _cameras_ts;_cameras_ts=0
_obs_queue=Queue(maxsize=200)
def save_observation(obs):
    try:_obs_queue.put_nowait(obs)
    except:pass  # drop if queue full

def _obs_writer():
    """Background thread that writes observations to DB without blocking analyzer."""
    while True:
        try:
            obs=_obs_queue.get(timeout=5)
            db_exec("INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (obs["id"],obs["camera_id"],obs["camera_name"],obs["timestamp"],obs["text"],
                 obs.get("is_alert",0),json.dumps(obs.get("alert_keywords",[])),
                 obs.get("snapshot_path",""),obs.get("thumb_b64",""),0))
        except Empty:continue
        except Exception as e:print(f"[DB] Write error: {e}")
threading.Thread(target=_obs_writer,daemon=True).start()
init_db()

# ===============================================================================
# TELEGRAM
# ===============================================================================
_last_tg=0
def send_telegram(text,photo_path=None):
    global _last_tg
    s=get_all_settings();token=s.get("telegram_token","");cid=s.get("telegram_chat_id","")
    if not token or not cid:return
    qs,qe=s.get("telegram_quiet_start",""),s.get("telegram_quiet_end","")
    if qs and qe:
        now_hm=datetime.now().strftime("%H:%M")
        if(qs<=qe and qs<=now_hm<=qe)or(qs>qe and(now_hm>=qs or now_hm<=qe)):return
    mi=sint(s.get("telegram_min_interval","30"),30)
    if time.time()-_last_tg<mi:return
    try:
        if photo_path and os.path.isfile(photo_path):
            with open(photo_path,"rb")as f:req.post(f"https://api.telegram.org/bot{token}/sendPhoto",data={"chat_id":cid,"caption":text[:1024],"parse_mode":"HTML"},files={"photo":f},timeout=15)
        else:req.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":cid,"text":text[:4096],"parse_mode":"HTML"},timeout=15)
        _last_tg=time.time()
    except Exception as e:print(f"[TG] {e}")

_last_ntfy=0
def send_ntfy(text,title="Mevin Alert",photo_path=None,priority="high",tags="rotating_light"):
    """Send a push notification via ntfy.sh - dead simple, no bot setup.
    User just installs the ntfy app and subscribes to their topic."""
    global _last_ntfy
    s=get_all_settings();topic=s.get("ntfy_topic","").strip()
    if not topic:return
    server=s.get("ntfy_server","https://ntfy.sh").strip().rstrip("/")
    mi=sint(s.get("ntfy_min_interval","30"),30)
    if time.time()-_last_ntfy<mi:return
    url=f"{server}/{topic}"
    try:
        headers={"Title":title[:200],"Priority":priority,"Tags":tags}
        if photo_path and os.path.isfile(photo_path):
            # ntfy: attach image by PUT with the file body
            with open(photo_path,"rb")as f:
                req.put(url,data=f.read(),headers={**headers,"Filename":"snapshot.jpg","Message":text[:500]},timeout=15)
        else:
            req.post(url,data=text[:1000].encode("utf-8"),headers=headers,timeout=15)
        _last_ntfy=time.time()
    except Exception as e:print(f"[ntfy] {e}")

def send_push(text,photo_path=None,is_alert=True):
    """Fan out a notification to all configured channels (Telegram + ntfy)."""
    send_telegram(text,photo_path)
    send_ntfy(text,title=("Mevin Alert" if is_alert else "Mevin"),photo_path=photo_path,
              priority=("high" if is_alert else "default"))

# ===============================================================================
# CAMERA FEED
# ===============================================================================
class CameraFeed:
    def __init__(s,cfg):
        s.id=cfg["id"];s.name=cfg["name"];s.source=cfg["source"];s.zone=cfg.get("zone","Main");s.frame=None;s.lock=threading.Lock();s.running=True;s.connected=False
        s.is_file=not str(s.source).isdigit() and not str(s.source).startswith(("rtsp://","http://","https://"))
        s.native_fps=25  # will be updated once connected
        threading.Thread(target=s._run,daemon=True).start()
    def _run(s):
        while s.running:
            src=int(s.source)if str(s.source).isdigit()else s.source
            cap=cv2.VideoCapture(src)
            if not cap.isOpened():s.connected=False;time.sleep(5);continue
            s.connected=True;cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            fps=cap.get(cv2.CAP_PROP_FPS)
            if fps<=0 or fps>120:fps=25
            s.native_fps=fps
            delay=1.0/fps if s.is_file else 0
            print(f"[{s.id}] Connected: {s.name} ({fps:.0f}fps{'  file' if s.is_file else ''})")
            while s.running:
                ok,f=cap.read()
                if not ok:
                    if s.is_file:cap.set(cv2.CAP_PROP_POS_FRAMES,0);continue
                    s.connected=False;break
                with s.lock:s.frame=f
                if s.is_file:time.sleep(delay)  # file: pace at native FPS
            cap.release()
    def get_frame(s):
        with s.lock:return s.frame.copy()if s.frame is not None else None
    def thumbnail(s,frame):
        h,w=frame.shape[:2];sc=THUMB_WIDTH/w;t=cv2.resize(frame,(THUMB_WIDTH,int(h*sc)),interpolation=cv2.INTER_AREA)
        _,jpg=cv2.imencode(".jpg",t,[cv2.IMWRITE_JPEG_QUALITY,70]);return base64.b64encode(jpg.tobytes()).decode()
    def _ensure_encoder(s):
        """Start one shared background encoder per camera. All viewers read its output -
        so N browser tabs / board tiles cost ONE encode, not N."""
        if getattr(s,"_enc_thread",None) and s._enc_thread.is_alive():return
        s._enc_lock=threading.Lock()
        s._enc_full=None   # latest full-size JPEG bytes (focus view)
        s._enc_small=None  # latest small JPEG bytes (board tiles)
        s._enc_viewers=0
        s._enc_last_view=time.time()
        def loop():
            while s.running:
                # Idle down when nobody is watching (saves CPU/GIL)
                idle=time.time()-s._enc_last_view>5
                rate=0.5 if idle else 1.0/STREAM_FPS
                f=s.get_frame()
                if f is not None:
                    h,w=f.shape[:2]
                    # Full: capped width for focus view
                    fw=MJPEG_WIDTH
                    ff=f if w<=fw else cv2.resize(f,(fw,int(h*fw/w)),interpolation=cv2.INTER_AREA)
                    ok1,j1=cv2.imencode(".jpg",ff,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
                    # Small: board tile thumbnail
                    sw=384
                    sf=f if w<=sw else cv2.resize(f,(sw,int(h*sw/w)),interpolation=cv2.INTER_AREA)
                    ok2,j2=cv2.imencode(".jpg",sf,[cv2.IMWRITE_JPEG_QUALITY,55])
                    with s._enc_lock:
                        if ok1:s._enc_full=j1.tobytes()
                        if ok2:s._enc_small=j2.tobytes()
                time.sleep(rate)
        s._enc_thread=threading.Thread(target=loop,daemon=True);s._enc_thread.start()
    def latest_jpeg(s,small=False):
        s._ensure_encoder();s._enc_last_view=time.time()
        with s._enc_lock:return s._enc_small if small else s._enc_full
    def mjpeg_gen(s,small=False):
        s._ensure_encoder()
        interval=1.0/STREAM_FPS
        while s.running:
            s._enc_last_view=time.time()
            b=s.latest_jpeg(small)
            if b is None:
                blank=np.zeros((360,640,3),dtype=np.uint8);_,jpg=cv2.imencode(".jpg",blank);b=jpg.tobytes()
            yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+b+b"\r\n"
            time.sleep(interval)
    def stop(s):s.running=False

# ===============================================================================
# GPU MONITOR
# ===============================================================================
class GPUMonitor:
    def __init__(s):s.data={"available":False,"name":"","vram_used":0,"vram_total":0,"vram_pct":0,"temp":0,"util":0};s.running=True;threading.Thread(target=s._loop,daemon=True).start()
    def _poll(s):
        try:
            r=subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5)
            if r.returncode!=0:return
            p=[x.strip()for x in r.stdout.strip().split(",")]
            if len(p)>=5:vu,vt,tp,ut=int(p[1]),int(p[2]),int(p[3]),int(p[4]);s.data={"available":True,"name":p[0],"vram_used":vu,"vram_total":vt,"vram_pct":round(vu/max(1,vt)*100,1),"temp":tp,"util":ut}
        except:pass
    def _loop(s):
        while s.running:s._poll();time.sleep(3)
    def to_dict(s):return dict(s.data)
    def stop(s):s.running=False
gpu_monitor=GPUMonitor()

# ===============================================================================
# SCENE MEMORY - per-camera situational understanding
# ===============================================================================
class SceneMemory:
    """A living, situational understanding of one camera.
    Tracks what's normal, what's been happening, and where danger is trending."""
    def __init__(s):
        s.baseline=""           # learned "normal" description for this camera
        s.history=deque(maxlen=6)  # recent (time_str, caption, danger_score)
        s.danger_trail=deque(maxlen=20)  # danger scores over time
        s.calm_count=0          # consecutive calm frames (for baseline learning)
        s.last_situation=""     # last situational summary

    def build_context(s,base_prompt,model=""):
        """Compose a situational prompt: baseline + recent narrative + the ask.
        Small models (moondream) get a compact version - they lose focus on long prompts."""
        small=any(m in model.lower() for m in["moondream","moon","1.6b","0.5b","tiny","nano"])
        if small:
            # Compact: one short line of baseline + the ask. No long history block.
            if s.baseline:
                return f"Normally: {s.baseline[:80]}. Now - {base_prompt} Note if anything changed or looks dangerous."
            return base_prompt
        # Full situational reasoning for capable models
        parts=[]
        if s.baseline:
            parts.append(f"NORMAL for this camera: {s.baseline}")
        if s.history:
            recent="\n".join(f"  [{t}] {c}" for t,c,_ in s.history)
            parts.append(f"WHAT JUST HAPPENED:\n{recent}")
        if s.baseline or s.history:
            parts.append(
                base_prompt+
                " Read the SITUATION, not just objects: what are people doing, what is their"
                " intent, how does this differ from normal, and is it escalating, stable, or"
                " calming. Call out crime or danger the moment you see it. One sentence."
            )
        else:
            parts.append(base_prompt)
        return "\n\n".join(parts)

    def trend(s):
        """Return danger trajectory: rising / falling / stable."""
        if len(s.danger_trail)<4:return"stable"
        recent=list(s.danger_trail)
        first=sum(recent[:len(recent)//2])/max(1,len(recent)//2)
        last=sum(recent[len(recent)//2:])/max(1,len(recent)-len(recent)//2)
        if last>first+0.15:return"rising"
        if last<first-0.15:return"falling"
        return"stable"

    def update(s,caption,danger,time_str):
        s.history.append((time_str,caption,danger))
        s.danger_trail.append(danger)
        s.last_situation=caption
        # Learn baseline from calm, stable moments
        if danger<0.1:
            s.calm_count+=1
            # After a few calm frames, this is "normal" for the camera
            if s.calm_count>=3 and(not s.baseline or s.calm_count%20==0):
                s.baseline=caption
        else:
            s.calm_count=0

# ===============================================================================
# VLM ANALYZER
# ===============================================================================
class VLMAnalyzer:
    def __init__(s):
        s.cameras={};s.sse_queues=[];s.sse_lock=threading.Lock();s.running=True;s.paused=False
        s.prev_caption={};s.prev_gray={};s.no_change_count={};s.focus_cam=None;s.frames_per_cam=3;s.cur_frame=0
        s.memory={}      # cam_id -> SceneMemory
        s.situation={}   # cam_id -> {text, danger, trend, time, is_alert} - the current pinned read
        s.zone_picture={}# zone -> {text, time, danger}
        s.zone_ts={}     # zone -> last synth time
        # Limit concurrent VLM calls so parallel workers don't thrash one GPU.
        s.vlm_slots=threading.Semaphore(sint(os.environ.get("MEVIN_GPU_SLOTS","1"),1))
        threading.Thread(target=s._loop,daemon=True).start()
        threading.Thread(target=s._zone_loop,daemon=True).start()
    def scene_memory(s,cid):
        if cid not in s.memory:s.memory[cid]=SceneMemory()
        return s.memory[cid]

    def _zone_loop(s):
        """Big-picture engine: periodically synthesize each zone's cameras into one
        overall understanding. Text-only (fast) - combines the live situation reads."""
        while s.running:
            time.sleep(6)
            if s.paused:continue
            if s._external_worker_alive():continue  # worker process handles zone synthesis
            try:
                st=get_all_settings()
                if st.get("zone_synthesis","true")!="true":continue
                # Group current situations by zone
                zones={}
                for cid,cam in list(s.cameras.items()):
                    sit=s.situation.get(cid)
                    if not sit:continue
                    z=getattr(cam,"zone","Main") or "Main"
                    zones.setdefault(z,[]).append((cam.name,sit))
                for zone,cams in zones.items():
                    # Only synthesize zones with 2+ cameras (a "circle" watching one area)
                    if len(cams)<2:continue
                    # Skip if nothing has changed recently in this zone
                    newest=max(c[1].get("ts",0) for c in cams)
                    if newest<=s.zone_ts.get(zone,0):continue
                    s.zone_ts[zone]=newest
                    s._synthesize_zone(zone,cams,st)
            except Exception as e:print(f"[zone] {e}")

    def _synthesize_zone(s,zone,cams,st):
        """Ask the model to combine multiple camera reads into one big-picture narrative."""
        reports="\n".join(f"  {name}: {sit.get('text','')}" for name,sit in cams)
        max_danger=max((sit.get("danger",0) for _,sit in cams),default=0)
        prompt=(f"These cameras all watch the same area ({zone}). Each reports what it sees:\n"
                f"{reports}\n\n"
                "In ONE sentence, describe the overall situation across this area. "
                "If a person or event appears to move between cameras, say so. "
                "Focus on the big picture, not each camera separately.")
        model=st.get("model","gemma3:4b")
        base_url=st.get("ollama_url",DEFAULTS["ollama_url"]).replace("/api/generate","").replace("/api/chat","").rstrip("/")
        s.vlm_slots.acquire()
        try:
            r=req.post(f"{base_url}/v1/chat/completions",json={
                "model":model,"messages":[{"role":"user","content":prompt}],
                "stream":False,"max_tokens":60,"temperature":0.3,"keep_alive":"30m",
            },timeout=60)
            txt=""
            if r.status_code<400:
                j=r.json();txt=(j.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
            if txt:
                s.zone_picture[zone]={"text":txt,"time":datetime.now().strftime("%H:%M:%S"),"danger":round(max_danger,2)}
                s._broadcast({"type":"zone_summary","zone":zone,"text":txt,
                    "time":datetime.now().strftime("%H:%M:%S"),"danger":round(max_danger,2),
                    "cameras":[name for name,_ in cams]})
        except Exception as e:print(f"[zone-synth] {e}")
        finally:s.vlm_slots.release()
    def set_cameras(s,d):s.cameras=d
    def set_focus(s,cid):s.focus_cam=cid;s.cur_frame=0
    def _broadcast(s,data):
        with s.sse_lock:
            dead=[]
            for q in s.sse_queues:
                try:q.put_nowait(data)
                except:dead.append(q)
            for q in dead:s.sse_queues.remove(q)
    def _motion(s,cid,frame):
        sm=cv2.resize(frame,(320,180),interpolation=cv2.INTER_AREA);g=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY);g=cv2.GaussianBlur(g,(21,21),0)
        prev=s.prev_gray.get(cid);s.prev_gray[cid]=g
        if prev is None:return 100.0
        d=cv2.absdiff(prev,g);_,th=cv2.threshold(d,25,255,cv2.THRESH_BINARY);return(np.count_nonzero(th)/(th.shape[0]*th.shape[1]))*100
    def _clean(s,t,prompt=""):
        import re
        t=t.strip()
        # Strip prompt echo - model sometimes repeats the prompt
        if prompt:
            # Remove exact prompt text
            t=t.replace(prompt,"").strip()
            # Remove quoted prompt fragments
            for frag in prompt.split('.'):
                frag=frag.strip()
                if len(frag)>10:
                    t=t.replace(frag,"").strip()
                    t=t.replace('"'+frag+'"',"").strip()
                    t=t.replace("'"+frag+"'",'').strip()
        # Strip thinking/analysis blocks
        think_markers=['Thinking Process','thinking process','Analyze the Request','Analyze the Image',
                       'Goal:','Output Format:','Required Content:','Step ','**Analyze','**Examine',
                       '**Setting:','**Subject','**Action','**Object','Answer the prompt']
        has_thinking=any(m in t for m in think_markers)
        if has_thinking:
            parts=[p.strip() for p in t.split('\n\n') if p.strip()]
            clean_parts=[]
            for p in reversed(parts):
                p_clean=re.sub(r'\*\*[^*]+\*\*:?\s*','',p).strip()
                if p_clean and not any(m.lower() in p_clean.lower() for m in ['analyze','thinking','goal:','format:','step ','required','output:','process','answer the prompt']):
                    clean_parts.insert(0,p_clean)
                    if len(clean_parts)>=2:break
            if clean_parts:t=' '.join(clean_parts)
            else:
                last=t.split('.')
                t='. '.join(last[-3:]).strip() if len(last)>3 else t
        # Strip markdown
        t=re.sub(r'\*\*([^*]+)\*\*',r'\1',t)
        t=re.sub(r'\*([^*]+)\*',r'\1',t)
        t=re.sub(r'^#+\s*','',t,flags=re.MULTILINE)
        t=re.sub(r'^\d+\.\s*','',t,flags=re.MULTILINE)
        t=re.sub(r'^\*\s+','',t,flags=re.MULTILINE)
        t=re.sub(r'\n{2,}','. ',t)
        t=re.sub(r'\n',' ',t)
        t=re.sub(r'\s{2,}',' ',t)
        # Strip preambles - loop until no more matches
        changed=True
        while changed:
            changed=False
            for p in["An image showing ","An image of ","The image shows ","The scene shows ",
                      "In this image, ","In this scene, ","I can see ","Here is ","Here's ",
                      "This appears ","This image shows ","Based on the image, ",
                      "The camera shows ","The image depicts ","In the image, ",
                      "Content: ","Analyze the Image: ","A wide shot of ","A shot of ",
                      "An analysis of the scene","An analysis of ","Analysis: ",
                      "Here is my analysis","Here is the analysis","Scene analysis:",
                      "Security analysis:","Scene description:","Summary: ",
                      "An outdoor ","A view of ","Looking at ","We can see ",
                      "What is happening? ","Observation: ",
                      "The updated image:.","The updated image:","Updated image:",
                      "The image:.","Image analysis:","Scene:",
                      "A description of the scene:.","A description of the scene:",
                      "A description of the scene","Description of the scene:",
                      "Description:","Here is a description"]:
                if t.lower().startswith(p.lower()):t=t[len(p):].strip();changed=True
        # Remove leftover quotes and colons at start
        t=re.sub(r'^["\':.\s]+','',t)
        if t and t[0].islower():t=t[0].upper()+t[1:]
        # Cap at 3 sentences
        sents=re.split(r'(?<=[.!?])\s+',t)
        if len(sents)>3:t=' '.join(sents[:3])
        return t.strip()
    def _analyze(s,cam):
        frame=cam.get_frame()
        if frame is None:return
        st=get_all_settings();mp=s._motion(cam.id,frame);ms=sfloat(st.get("motion_sensitivity","0.3"),0.3)
        if st.get("motion_enabled","true")=="true":
            nc=s.no_change_count.get(cam.id,0)
            if mp<ms and nc>0 and nc<5:s.no_change_count[cam.id]=nc+1;return
        bp=st.get("prompt",DEFAULTS["prompt"])
        mem=s.scene_memory(cam.id)
        situational=st.get("situational","true")=="true"
        model=st.get("model","gemma3:4b")
        ctx=mem.build_context(bp,model) if situational else (bp if not s.prev_caption.get(cam.id,"") else f"{bp}\nPrevious: \"{s.prev_caption.get(cam.id,'')}\"")
        caption=""
        base_url=st.get("ollama_url",DEFAULTS["ollama_url"]).replace("/api/generate","").replace("/api/chat","").rstrip("/")
        s.vlm_slots.acquire()  # wait for a free GPU slot (serializes inference on 1 GPU)
        # Real-time: grab the FRESHEST frame now the GPU is ours, not the one from before the wait
        fresh=cam.get_frame()
        if fresh is not None:frame=fresh;mp=s._motion(cam.id,frame)
        ts=datetime.now();iid=f"{cam.id}_{int(ts.timestamp()*1000)}"
        thumb=cam.thumbnail(frame)
        h,w=frame.shape[:2]
        isz=sint(st.get("inference_size","640"),640)
        sc=isz/max(h,w)
        sm=cv2.resize(frame,(int(w*sc),int(h*sc)),interpolation=cv2.INTER_AREA) if sc<1 else frame
        _,jpg=cv2.imencode(".jpg",sm,[cv2.IMWRITE_JPEG_QUALITY,92])
        b64=base64.b64encode(jpg.tobytes()).decode()
        print(f"[{cam.name}] motion={mp:.1f}% analyzing (live)...")
        s._broadcast({"type":"stream_start","camera_id":cam.id,"item_id":iid,"camera_name":cam.name,"time":ts.strftime("%H:%M:%S"),"thumb":thumb,"motion":round(mp,1)})
        try:
            r=req.post(f"{base_url}/v1/chat/completions",json={
                "model":model,
                "messages":[{"role":"user","content":[
                    {"type":"text","text":ctx},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]}],
                "stream":True,"max_tokens":sint(st.get("max_tokens","150"),150),"temperature":0.2,
                "keep_alive":"30m",
            },headers={"Content-Type":"application/json"},stream=True,timeout=120)
            if r.status_code>=400:
                try:err=r.json().get("error",{});emsg=err.get("message","") if isinstance(err,dict) else str(err)
                except:emsg=r.text[:300]
                print(f"  [API] error ({r.status_code}): {emsg}")
                # Model not installed -> surface a clear instruction in the feed
                if "not found" in emsg.lower() or "no such model" in emsg.lower() or "try pulling" in emsg.lower():
                    hint=f"Model '{model}' is not installed. Run: ollama pull {model}"
                    s._broadcast({"type":"stream_token","item_id":iid,"text":hint})
                    caption=hint;raise RuntimeError(hint)
                r=req.post(f"{base_url}/api/chat",json={"model":model,"messages":[{"role":"user","content":ctx,"images":[b64]}],"stream":True,"keep_alive":"30m","options":{"temperature":0.2,"num_predict":sint(st.get("max_tokens","150"),150)}},stream=True,timeout=120)
            r.raise_for_status()
            thinking="";content_started=False
            for line in r.iter_lines():
                if not s.running:break
                if not line:continue
                line_str=line.decode("utf-8",errors="ignore") if isinstance(line,bytes) else line
                if line_str.startswith("data: "):line_str=line_str[6:]
                if line_str.strip()=="[DONE]":break
                try:
                    chunk=json.loads(line_str)
                    tk=""
                    if "choices" in chunk:
                        delta=chunk["choices"][0].get("delta",{})
                        tk=delta.get("content","") or ""
                    else:
                        tk=chunk.get("response","") or chunk.get("message",{}).get("content","") or ""
                    th=chunk.get("message",{}).get("thinking","") or chunk.get("thinking","") or ""
                    if th:thinking+=th;continue
                    if tk:
                        if not content_started:
                            test=(caption+tk).lower()
                            if any(m in test for m in ["analyze the","thinking process","goal:","output format","step 1","**analyze"]):
                                caption+=tk;continue
                            content_started=True;caption=tk
                        else:caption+=tk
                        s._broadcast({"type":"stream_token","item_id":iid,"text":s._clean(caption,bp)})
                except:continue
            if not caption.strip() and thinking.strip():caption=thinking
        except RuntimeError:pass  # already surfaced a clean message (e.g. model not installed)
        except Exception as e:caption=f"Error: {e}"
        finally:s.vlm_slots.release()  # free the GPU slot for the next camera
        caption=s._clean(caption.strip(),bp) or "(no response)"
        no_chg=caption.lower().strip().startswith("no change")
        if no_chg:s.no_change_count[cam.id]=s.no_change_count.get(cam.id,0)+1
        else:s.no_change_count[cam.id]=0;s.prev_caption[cam.id]=caption
        if cam.id not in s.prev_caption:s.prev_caption[cam.id]=caption
        kws=json.loads(st.get("alert_keywords","[]"))
        cap_low=caption.lower();triggered=[]
        for k in kws:
            kl=k.lower()
            if kl not in cap_low:continue
            pos=0;found_positive=False
            while True:
                idx=cap_low.find(kl,pos)
                if idx<0:break
                window=cap_low[max(0,idx-50):idx]
                if not _NEG_RE.search(window):found_positive=True;break
                pos=idx+1
            if found_positive:triggered.append(k)
        is_alert=len(triggered)>0
        # Compute danger score for the situational trajectory
        danger=0.02 if no_chg else 0.1
        if is_alert:danger=0.5+min(len(triggered)*0.12,0.4)
        danger+=min(mp/100,1)*0.1
        danger=min(1.0,danger)
        # Update the living scene memory
        trend="stable"
        if situational and not no_chg:
            mem.update(caption,danger,ts.strftime("%H:%M:%S"))
            trend=mem.trend()
        # Store the current pinned situation for this camera (always-visible read + zone synthesis)
        if not no_chg:
            s.situation[cam.id]={"text":caption,"danger":round(danger,2),"trend":trend,
                "time":ts.strftime("%H:%M:%S"),"is_alert":is_alert,"ts":time.time()}
        snap=""
        if is_alert or st.get("snapshot_on_every","false")=="true":
            dd=os.path.join(SNAPSHOT_DIR,ts.strftime("%Y-%m-%d"));os.makedirs(dd,exist_ok=True)
            sn=f"{cam.id}_{ts.strftime('%H%M%S')}.jpg";snap=os.path.join(dd,sn);cv2.imwrite(snap,frame,[cv2.IMWRITE_JPEG_QUALITY,sint(st.get("snapshot_quality","90"),90)])
        if not no_chg or is_alert:save_observation({"id":iid,"camera_id":cam.id,"camera_name":cam.name,"timestamp":ts.isoformat(),"text":caption,"is_alert":int(is_alert),"alert_keywords":triggered,"snapshot_path":snap,"thumb_b64":thumb})
        s._broadcast({"type":"feed_item","item_id":iid,"camera_id":cam.id,"camera_name":cam.name,"time":ts.strftime("%H:%M:%S"),
            "text":caption,"thumb":thumb,"is_alert":is_alert,"alert_keywords":triggered,
            "no_change":no_chg,"motion":round(mp,1),"snapshot":snap.replace("\\","/") if snap else"","trend":trend,"danger":round(danger,2)})
        notify_alert=is_alert and st.get("telegram_on_alert","true")=="true"
        notify_all=st.get("telegram_on_all","false")=="true" and not no_chg
        if notify_alert:threading.Thread(target=send_push,args=(f"ALERT {cam.name} {ts.strftime('%H:%M:%S')}\n{caption}\nKeywords: {', '.join(triggered)}",snap,True),daemon=True).start()
        elif notify_all:threading.Thread(target=send_push,args=(f"{cam.name} {ts.strftime('%H:%M:%S')}\n{caption}",None,False),daemon=True).start()
    def _loop(s):
        """Dispatcher: keeps one analysis worker per connected camera running in parallel.
        Cameras no longer wait in line - each is analyzed on its own thread."""
        s.last_analyzed={}      # cam_id -> timestamp
        s.analysis_times={}     # cam_id -> last duration (s)
        s.workers={}            # cam_id -> Thread
        s.load_warned=False
        # GPU/status broadcaster runs independently so the UI stays live
        threading.Thread(target=s._status_loop,daemon=True).start()
        while s.running:
            if s.paused:time.sleep(0.5);continue
            cams=list(s.cameras.values())
            # Focus mode: only the focused camera is analyzed
            if s.focus_cam and s.focus_cam in s.cameras:
                active=[s.cameras[s.focus_cam]]
            else:
                active=[c for c in cams if c.connected]
            # Spawn a worker for any active camera that doesn't have one
            for cam in active:
                w=s.workers.get(cam.id)
                if w is None or not w.is_alive():
                    t=threading.Thread(target=s._camera_worker,args=(cam.id,),daemon=True)
                    s.workers[cam.id]=t;t.start()
            time.sleep(0.5)

    def _external_worker_alive(s):
        """True if a separate analysis worker process posted a fresh heartbeat (<10s)."""
        try:
            hb=get_all_settings().get("_worker_heartbeat","")
            if not hb:return False
            return (time.time()-float(hb))<10
        except:return False

    def _camera_worker(s,cam_id):
        """Analyze one camera continuously, paced by the interval. One per camera.
        Yields to an external worker process if one is alive (heartbeat fresh)."""
        while s.running:
            if s.paused:time.sleep(0.5);continue
            cam=s.cameras.get(cam_id)
            if cam is None or not cam.connected:return
            if s.focus_cam and s.focus_cam!=cam_id:return
            # If a separate worker process is handling analysis, stand down (auto-resume if it dies)
            if s._external_worker_alive():time.sleep(2);continue
            t0=time.time()
            try:s._analyze(cam)
            except Exception as e:print(f"[{cam_id}] analyze error: {e}")
            s.analysis_times[cam_id]=time.time()-t0
            s.last_analyzed[cam_id]=time.time()
            # Cadence = max(interval, analysis time): back-to-back on fresh frames when
            # GPU-bound, exact interval when the GPU has spare capacity.
            st=get_all_settings();interval=max(0.2,sfloat(st.get("analysis_interval","3"),3))
            time.sleep(max(0.0,interval-(time.time()-t0)))

    def _status_loop(s):
        """Broadcast GPU + rotation status on a steady cadence, independent of analysis."""
        while s.running:
            try:
                gpu=gpu_monitor.to_dict()
                cams=list(s.cameras.values())
                n_cams=len([c for c in cams if c.connected])
                if s.paused:
                    s._broadcast({"type":"gpu","data":gpu,"paused":True,"rotation":{"current":"Paused","total_cameras":n_cams,"cycle_time":0,"focus":s.focus_cam}})
                    time.sleep(1.5);continue
                # With parallel workers, "cycle time" is the slowest single camera, not the sum
                slowest=max(s.analysis_times.values()) if s.analysis_times else 0
                status={"type":"gpu","data":gpu,"rotation":{
                    "current":"Parallel" if n_cams>1 and not s.focus_cam else (s.cameras[s.focus_cam].name if s.focus_cam and s.focus_cam in s.cameras else (cams[0].name if cams else "-")),
                    "total_cameras":n_cams,"cycle_time":round(slowest,1),
                    "per_camera":{cid:round(dt,1) for cid,dt in s.analysis_times.items()},
                    "focus":s.focus_cam}}
                warns=[];overloaded=False
                if gpu.get("available"):
                    if gpu["vram_pct"]>90:warns.append(f"VRAM at {gpu['vram_pct']}%");overloaded=True
                    if gpu["temp"]>82:warns.append(f"GPU temp {gpu['temp']}degC");overloaded=True
                if slowest>30:warns.append(f"Analysis taking {slowest:.0f}s/frame");overloaded=True
                if overloaded and not s.load_warned:
                    status["load_warning"]="; ".join(warns);print(f"[LOAD] {'; '.join(warns)}");s.load_warned=True
                elif not overloaded:s.load_warned=False
                s._broadcast(status)
            except Exception as e:print(f"[status] {e}")
            time.sleep(1.5)
    def subscribe(s):
        q=Queue(maxsize=20)  # small queue prevents memory buildup
        with s.sse_lock:s.sse_queues.append(q)
        return q
    def unsubscribe(s,q):
        with s.sse_lock:
            if q in s.sse_queues:s.sse_queues.remove(q)
    def stop(s):s.running=False

analyzer=VLMAnalyzer();feeds={}
def start_cameras():
    global feeds
    for c in feeds.values():c.stop()
    feeds.clear()
    for cam in get_cameras_from_db():feeds[cam["id"]]=CameraFeed(cam)
    analyzer.set_cameras(feeds)

def add_single_camera(cam_dict):
    global feeds
    feeds[cam_dict["id"]]=CameraFeed(cam_dict)
    analyzer.set_cameras(feeds);invalidate_cameras()

def remove_single_camera(cid):
    global feeds
    if cid in feeds:feeds[cid].stop();del feeds[cid]
    analyzer.set_cameras(feeds);invalidate_cameras()

start_cameras()

# ===============================================================================
# DATA MAINTENANCE
# ===============================================================================
class DataMaintenance:
    def __init__(s):s.running=True;s.last_stats={};threading.Thread(target=s._loop,daemon=True).start()
    def _db_size(s):
        try:return os.path.getsize(DB_PATH)/(1024*1024)
        except:return 0
    def _snap_size(s):
        t=0
        for root,_,files in os.walk(SNAPSHOT_DIR):
            for f in files:
                try:t+=os.path.getsize(os.path.join(root,f))
                except:pass
        return t/(1024*1024)
    def _count_obs(s):
        try:return db_exec("SELECT COUNT(*) FROM observations",fetchone=True)[0]
        except:return 0
    def run_cleanup(s):
        st=get_all_settings();rd=sint(st.get("retain_days","30"),30);mo=sint(st.get("retain_max_obs","5000"),5000)
        msm=sint(st.get("retain_max_snap_mb","2000"),2000);td=sint(st.get("thumb_retain_days","7"),7)
        cutoff=(datetime.now()-timedelta(days=rd)).isoformat();tcutoff=(datetime.now()-timedelta(days=td)).isoformat()
        db_exec("DELETE FROM observations WHERE timestamp<? AND pinned=0",(cutoff,))
        count=db_exec("SELECT COUNT(*) FROM observations WHERE pinned=0",fetchone=True)[0]
        if count>mo:db_exec("DELETE FROM observations WHERE id IN (SELECT id FROM observations WHERE pinned=0 ORDER BY timestamp ASC LIMIT ?)",(count-mo,))
        db_exec("UPDATE observations SET thumb_b64='' WHERE timestamp<? AND thumb_b64!='' AND pinned=0",(tcutoff,))
        try:
            with _db_lock:get_shared_db().execute("VACUUM")
        except:pass
        # Snapshot cleanup
        cutoff_dt=datetime.now()-timedelta(days=rd)
        for root,dirs,files in os.walk(SNAPSHOT_DIR,topdown=False):
            for f in files:
                fp=os.path.join(root,f)
                try:
                    if datetime.fromtimestamp(os.path.getmtime(fp))<cutoff_dt:os.remove(fp)
                except:pass
            try:
                if not os.listdir(root)and root!=SNAPSHOT_DIR:os.rmdir(root)
            except:pass
        # Size cap
        if s._snap_size()>msm:
            all_f=[];
            for root,_,files in os.walk(SNAPSHOT_DIR):
                for f in files:fp=os.path.join(root,f);
                try:all_f.append((os.path.getmtime(fp),fp))
                except:pass
            all_f.sort()
            for _,fp in all_f:
                if s._snap_size()<=msm*0.8:break
                try:os.remove(fp)
                except:pass
        s.last_stats={"db_size_mb":round(s._db_size(),1),"snap_size_mb":round(s._snap_size(),1),"total_obs":s._count_obs(),"retain_days":rd,"max_obs":mo,"max_snap_mb":msm}
    def _loop(s):
        time.sleep(10)
        while s.running:
            try:s.run_cleanup()
            except Exception as e:print(f"[Maintenance] {e}")
            st=get_all_settings();time.sleep(max(300,sint(st.get("cleanup_interval_min","60"),60)*60))
    def stop(s):s.running=False
maintenance=DataMaintenance()

# ===============================================================================
# SCANNERS
# ===============================================================================
def scan_usb(mx=10):
    found=[];old_log=os.environ.get("OPENCV_LOG_LEVEL","")
    os.environ["OPENCV_LOG_LEVEL"]="SILENT"
    for i in range(mx):
        cap=cv2.VideoCapture(i,cv2.CAP_DSHOW)  # DSHOW is quieter on Windows
        if cap.isOpened():w=int(cap.get(3));h=int(cap.get(4));found.append({"source":str(i),"name":f"USB Camera {i}","type":"usb","info":f"{w}x{h}"});cap.release()
        else:
            cap.release()
            if i>1 and not found:break  # stop early if nothing found after index 1
    if old_log:os.environ["OPENCV_LOG_LEVEL"]=old_log
    else:os.environ.pop("OPENCV_LOG_LEVEL",None)
    return found
def scan_network():
    try:s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(("8.8.8.8",80));lip=s.getsockname()[0];s.close()
    except:return[],""
    base=".".join(lip.split(".")[:3]);results=[];threads=[]
    def chk(ip):
        for port in[554,8554,8080]:
            try:sk=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sk.settimeout(0.4);r=sk.connect_ex((ip,port));sk.close()
            except:continue
            if r==0:
                try:h=socket.gethostbyaddr(ip)[0]
                except:h=""
                src=f"rtsp://admin:admin@{ip}:{port}/stream1"if port in(554,8554)else f"http://{ip}:{port}/video"
                results.append({"source":src,"name":h or f"Device {ip}","type":"rtsp"if port in(554,8554)else"http","info":f"port {port}"});break
    for i in range(1,255):
        ip=f"{base}.{i}"
        if ip!=lip:t=threading.Thread(target=chk,args=(ip,),daemon=True);threads.append(t);t.start()
    for t in threads:t.join(timeout=2)
    return results,lip
def scan_video_files():
    home=pathlib.Path.home()
    dirs=[home/"Downloads",home/"Videos",home/"Desktop",home/"Documents",pathlib.Path(".")]
    exts={'.mp4','.avi','.mkv','.mov','.wmv','.flv','.webm','.m4v'}
    found=[]
    for d in dirs:
        if not d.exists():continue
        try:
            for f in sorted(d.iterdir()):
                if f.is_file()and f.suffix.lower()in exts:
                    sz=round(f.stat().st_size/(1024*1024),1)
                    found.append({"path":str(f),"name":f.name,"dir":str(d),"size":f"{sz} MB"})
        except:pass
    return found[:50]

_scan={"status":"idle","usb":[],"network":[],"lip":""}
_scan_lock=threading.Lock()

# ===============================================================================
# HTML (same frontend, loaded from separate variable for clarity)
# ===============================================================================
# Load dashboard from same directory as script
_script_dir=os.path.dirname(os.path.abspath(__file__))
_html_path=os.path.join(_script_dir,"dashboard.html")
if os.path.exists(_html_path):
    with open(_html_path,encoding="utf-8") as f:HTML=f.read()
else:HTML='<!DOCTYPE html><html><head><title>Mevin</title></head><body style="background:#0f1117;color:#eee;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center"><h1>Mevin</h1><p>Place dashboard.html next to mevin.py</p></div></body></html>'

AUTH_TOKEN=os.environ.get("MEVIN_TOKEN","")
MJPEG_WIDTH=640

# ===============================================================================
# FASTAPI ROUTES
# ===============================================================================
from fastapi import Header,HTTPException,Query,Depends,APIRouter

async def check_auth(authorization:str=Header(default=""),token:str=Query(default="")):
    if not AUTH_TOKEN:return
    tok=token or authorization.replace("Bearer ","")
    if tok!=AUTH_TOKEN:raise HTTPException(401,"Invalid or missing token")

# Public routes (dashboard, video, SSE) - no auth
# API routes - auth protected when MEVIN_TOKEN is set
api=APIRouter(prefix="/api",dependencies=[Depends(check_auth)])

@app.get("/",response_class=HTMLResponse)
async def index():return HTML

@app.get("/video/{cam_id}")
async def video_feed(cam_id:str,small:int=0):
    cam=feeds.get(cam_id)
    if not cam:return Response(status_code=404)
    return StreamingResponse(cam.mjpeg_gen(small=bool(small)),media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/snapshot-live/{cam_id}")
async def snapshot_live(cam_id:str):
    """Single current JPEG from the shared encoder. Board tiles poll this every ~1.5s
    instead of holding a live stream - far less load, looks live, frees the GIL."""
    cam=feeds.get(cam_id)
    if not cam:return Response(status_code=404)
    b=cam.latest_jpeg(small=True)
    if b is None:
        blank=np.zeros((216,384,3),dtype=np.uint8);_,jpg=cv2.imencode(".jpg",blank);b=jpg.tobytes()
    return Response(content=b,media_type="image/jpeg",headers={"Cache-Control":"no-store"})

# -- Internal endpoints for the analysis worker process ------------------------
@app.get("/internal/frame-hq/{cam_id}")
async def internal_frame_hq(cam_id:str):
    """High-quality full frame for the worker to analyze. Worker fetches this instead
    of opening its own camera connection, so cameras stay solely owned here."""
    cam=feeds.get(cam_id)
    if not cam:return Response(status_code=404)
    f=cam.get_frame()
    if f is None:return Response(status_code=503)
    _,jpg=cv2.imencode(".jpg",f,[cv2.IMWRITE_JPEG_QUALITY,92])
    return Response(content=jpg.tobytes(),media_type="image/jpeg",headers={"Cache-Control":"no-store"})

@app.get("/internal/cameras")
async def internal_cameras():
    """Camera list + zones for the worker (id, name, zone, connected)."""
    return [{"id":c.id,"name":c.name,"zone":getattr(c,"zone","Main"),"connected":c.connected}
            for c in feeds.values()]

@app.post("/internal/event")
async def internal_event(req:Request):
    """Worker posts analysis events here. We relay to SSE clients and update state."""
    m=await req.json();t=m.get("type")
    if t=="feed_item":
        # Update live situation + zone state, persist observation
        cid=m.get("camera_id")
        if not m.get("no_change"):
            analyzer.situation[cid]={"text":m.get("text",""),"danger":m.get("danger",0),
                "trend":m.get("trend","stable"),"time":m.get("time",""),
                "is_alert":m.get("is_alert",False),"ts":time.time()}
        if m.get("_save"):
            save_observation({"id":m["item_id"],"camera_id":cid,"camera_name":m.get("camera_name",""),
                "timestamp":m.get("iso",datetime.now().isoformat()),"text":m.get("text",""),
                "is_alert":int(bool(m.get("is_alert"))),"alert_keywords":m.get("alert_keywords",[]),
                "snapshot_path":m.get("snapshot",""),"thumb_b64":m.get("thumb","")})
    elif t=="zone_summary":
        analyzer.zone_picture[m["zone"]]={"text":m.get("text",""),"time":m.get("time",""),"danger":m.get("danger",0)}
    # Relay to all dashboard SSE clients
    analyzer._broadcast(m)
    return {"ok":True}

@app.get("/events")
async def sse_events():
    q=analyzer.subscribe()
    async def gen():
        loop=asyncio.get_event_loop()
        try:
            while True:
                try:
                    msg=await asyncio.wait_for(loop.run_in_executor(None,lambda:q.get(timeout=0.5)),timeout=1)
                    yield f"data:{json.dumps(msg)}\n\n"
                except(Empty,asyncio.TimeoutError):
                    yield ":\n\n"
                await asyncio.sleep(0)  # yield to event loop
        except asyncio.CancelledError:pass
        except GeneratorExit:pass
        finally:analyzer.unsubscribe(q)
    return StreamingResponse(gen(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"})

@app.get("/snapshot/{file_path:path}")
async def snapshot_file(file_path:str):
    safe=os.path.normpath(file_path)
    if ".." in safe:return Response(status_code=403)
    p=os.path.join(SNAPSHOT_DIR,safe)
    if not os.path.isfile(p):return Response(status_code=404)
    with open(p,"rb")as f:data=f.read()
    return Response(content=data,media_type="image/jpeg")

# -- Camera CRUD ---------------------------------------------------------------
@api.get("/cameras")
async def get_cameras():return get_cameras_from_db()

@api.post("/cameras")
async def add_camera(req:Request):
    d=await req.json();cid=f"cam{int(time.time()*1000)}";zone=d.get("zone","Main")or"Main"
    db_exec("INSERT INTO cameras(id,name,source,zone)VALUES(?,?,?,?)",(cid,d["name"],d["source"],zone))
    add_single_camera({"id":cid,"name":d["name"],"source":d["source"],"zone":zone})
    return{"ok":True,"id":cid}

@api.put("/cameras/{cid}")
async def update_camera(cid:str,req:Request):
    d=await req.json();zone=d.get("zone","Main")or"Main"
    db_exec("UPDATE cameras SET name=?,source=?,zone=? WHERE id=?",(d["name"],d["source"],zone,cid))
    remove_single_camera(cid)
    add_single_camera({"id":cid,"name":d["name"],"source":d["source"],"zone":zone})
    return{"ok":True}

@api.delete("/cameras/{cid}")
async def delete_camera(cid:str):
    db_exec("DELETE FROM cameras WHERE id=?",(cid,))
    remove_single_camera(cid)
    return{"ok":True}

@api.post("/take-snapshot/{cid}")
async def take_snapshot(cid:str):
    cam=feeds.get(cid)
    if not cam:return{"ok":False}
    frame=cam.get_frame()
    if frame is None:return{"ok":False}
    ts=datetime.now();dd=os.path.join(SNAPSHOT_DIR,ts.strftime("%Y-%m-%d"));os.makedirs(dd,exist_ok=True)
    cv2.imwrite(os.path.join(dd,f"{cid}_manual_{ts.strftime('%H%M%S')}.jpg"),frame,[cv2.IMWRITE_JPEG_QUALITY,sint(get_setting("snapshot_quality"),90)])
    return{"ok":True}

# -- Settings ------------------------------------------------------------------
@api.get("/settings")
async def get_settings():return get_all_settings()

@api.post("/settings")
async def save_settings(req:Request):
    d=await req.json()
    for k,v in d.items():set_setting(k,v)
    return{"ok":True}

# -- Focus ---------------------------------------------------------------------
@api.get("/focus")
async def get_focus():return{"focus":analyzer.focus_cam}

@api.post("/focus")
async def set_focus(req:Request):
    d=await req.json();cid=d.get("camera_id");analyzer.set_focus(cid)
    set_setting("focus_cam",cid or "")  # so the worker process sees it
    return{"ok":True}

@api.get("/analysis-status")
async def analysis_status():return{"paused":analyzer.paused}

@api.post("/pause")
async def pause_analysis():
    analyzer.paused=True;set_setting("paused","true");print("[Analysis] Paused");return{"ok":True,"paused":True}

@api.post("/resume")
async def resume_analysis():
    analyzer.paused=False;set_setting("paused","false");print("[Analysis] Resumed");return{"ok":True,"paused":False}

# -- GPU -----------------------------------------------------------------------
@api.get("/gpu")
async def get_gpu():return gpu_monitor.to_dict()

# -- Observations --------------------------------------------------------------
@api.post("/pin/{oid}")
async def pin_obs(oid:str):
    db_exec("UPDATE observations SET pinned=1 WHERE id=?",(oid,))
    return{"ok":True}

@api.post("/unpin/{oid}")
async def unpin_obs(oid:str):
    db_exec("UPDATE observations SET pinned=0 WHERE id=?",(oid,))
    return{"ok":True}

@api.post("/clear")
async def clear_obs():
    db_exec("DELETE FROM observations")
    return{"ok":True}

# -- Feed Cache ----------------------------------------------------------------
@api.get("/recent-feed")
async def recent_feed(limit:int=50,camera_id:str=""):
    """Load recent observations from DB. Optionally filter to one camera."""
    if camera_id:
        rows=db_exec(
            "SELECT id,camera_id,camera_name,timestamp,text,is_alert,alert_keywords,thumb_b64,pinned "
            "FROM observations WHERE camera_id=? ORDER BY timestamp DESC LIMIT ?",(camera_id,limit),fetch=True)
    else:
        rows=db_exec(
            "SELECT id,camera_id,camera_name,timestamp,text,is_alert,alert_keywords,thumb_b64,pinned "
            "FROM observations ORDER BY timestamp DESC LIMIT ?",(limit,),fetch=True)
    items=[]
    for r in rows:
        ts=r["timestamp"]
        try:t=datetime.fromisoformat(ts).strftime("%H:%M:%S")
        except:t=ts
        items.append({"item_id":r["id"],"camera_id":r["camera_id"],"camera_name":r["camera_name"],
            "time":t,"text":r["text"],"is_alert":r["is_alert"],
            "alert_keywords":json.loads(r["alert_keywords"]or"[]"),
            "thumb":r["thumb_b64"]or"","pinned":r["pinned"]})
    return items

@api.get("/situations")
async def situations():
    """Current pinned situation for every camera + zone big-pictures. For page load."""
    return {"cameras":analyzer.situation,"zones":analyzer.zone_picture}

@api.get("/zones")
async def zones():
    """List zones and which cameras belong to each."""
    cams=get_cameras_from_db();z={}
    for c in cams:z.setdefault(c.get("zone","Main")or"Main",[]).append({"id":c["id"],"name":c["name"]})
    return z

# -- Timeline ------------------------------------------------------------------
@api.get("/timeline")
async def timeline(hours:int=12):
    cutoff=(datetime.now()-timedelta(hours=hours)).isoformat()
    rows=db_exec("SELECT id,camera_name,timestamp,text,is_alert,alert_keywords,pinned FROM observations WHERE timestamp>? ORDER BY timestamp ASC LIMIT 500",(cutoff,),fetch=True)
    return{"events":[{"id":r["id"],"cam":r["camera_name"],"ts":r["timestamp"],"text":r["text"],"alert":r["is_alert"],"kws":json.loads(r["alert_keywords"]or"[]"),"pin":r["pinned"]}for r in rows],"hours":hours}

# -- Gallery -------------------------------------------------------------------
@api.get("/gallery")
async def gallery():
    items=[]
    for root,_,files in os.walk(SNAPSHOT_DIR):
        for f in sorted(files,reverse=True):
            if f.endswith(".jpg"):
                rel=os.path.relpath(os.path.join(root,f),SNAPSHOT_DIR).replace("\\","/")
                parts=f.replace(".jpg","").split("_");date=os.path.basename(root);t=parts[-1]if parts else""
                items.append({"path":rel,"name":parts[0],"date":date,"time":f"{t[:2]}:{t[2:4]}:{t[4:6]}"if len(t)>=6 else t})
    return items[:200]

# -- Telegram ------------------------------------------------------------------
@api.post("/test-telegram")
async def test_telegram():
    s=get_all_settings();token=s.get("telegram_token","");cid=s.get("telegram_chat_id","")
    if not token or not cid:return{"ok":False,"error":"Missing token or chat ID"}
    try:
        r=req.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":cid,"text":"Mevin connected."},timeout=10)
        return{"ok":r.status_code==200}
    except Exception as e:return{"ok":False,"error":str(e)}

@api.post("/test-ntfy")
async def test_ntfy():
    s=get_all_settings();topic=s.get("ntfy_topic","").strip()
    if not topic:return{"ok":False,"error":"Enter an ntfy topic first"}
    server=s.get("ntfy_server","https://ntfy.sh").strip().rstrip("/")
    try:
        r=req.post(f"{server}/{topic}",data=b"Mevin connected. You'll get alerts here.",
            headers={"Title":"Mevin","Tags":"white_check_mark"},timeout=10)
        return{"ok":r.status_code<400,"topic":topic,"server":server}
    except Exception as e:return{"ok":False,"error":str(e)}

# -- ONVIF auto-discovery ------------------------------------------------------
_onvif={"status":"idle","devices":[]}
_onvif_lock=threading.Lock()

@api.get("/onvif-available")
async def onvif_available():
    return{"available":_ONVIF_OK}

@api.post("/onvif-discover")
async def onvif_discover_start():
    if not _ONVIF_OK:
        return{"ok":False,"error":"ONVIF support not installed. Run: pip install onvif-zeep wsdiscovery"}
    def do():
        global _onvif
        with _onvif_lock:_onvif={"status":"discovering","devices":[]}
        devs=onvif_discover(timeout=5)
        with _onvif_lock:_onvif={"status":"done","devices":devs}
    threading.Thread(target=do,daemon=True).start()
    return{"ok":True}

@api.get("/onvif-discover")
async def onvif_discover_status():
    with _onvif_lock:return dict(_onvif)

@api.post("/onvif-connect")
async def onvif_connect(req_:Request):
    """Given a device IP/port + credentials, return its RTSP streams (channels)."""
    d=await req_.json()
    res=onvif_streams(d.get("ip",""),sint(str(d.get("port",80)),80),
                      d.get("user","admin"),d.get("password",""))
    return res

@api.post("/onvif-add-all")
async def onvif_add_all(req_:Request):
    """Add every channel from an ONVIF device as cameras in one click."""
    d=await req_.json()
    res=onvif_streams(d.get("ip",""),sint(str(d.get("port",80)),80),
                      d.get("user","admin"),d.get("password",""))
    if not res.get("ok"):return res
    zone=d.get("zone","") or d.get("ip","Cameras")
    added=[]
    for st_ in res["streams"]:
        cid=f"cam{int(time.time()*1000)}{len(added)}"
        nm=st_["name"] if len(res["streams"])>1 else (d.get("name") or st_["name"])
        db_exec("INSERT INTO cameras(id,name,source,zone)VALUES(?,?,?,?)",(cid,nm,st_["url"],zone))
        add_single_camera({"id":cid,"name":nm,"source":st_["url"],"zone":zone})
        added.append({"id":cid,"name":nm})
        time.sleep(0.001)
    return{"ok":True,"added":added,"zone":zone}

# -- Scanner -------------------------------------------------------------------
@api.post("/scan")
async def start_scan():
    def do():
        global _scan
        with _scan_lock:_scan={"status":"scanning USB...","usb":[],"network":[],"lip":""}
        usb=scan_usb()
        with _scan_lock:_scan["usb"]=usb;_scan["status"]="scanning network..."
        net,lip=scan_network()
        with _scan_lock:_scan["network"]=net;_scan["lip"]=lip;_scan["status"]="done"
    threading.Thread(target=do,daemon=True).start();return{"ok":True}

@api.get("/scan")
async def scan_status():
    with _scan_lock:return dict(_scan)

@api.get("/video-files")
async def video_files():return scan_video_files()

# -- Data Stats ----------------------------------------------------------------
@api.get("/data-stats")
async def data_stats():return maintenance.last_stats

@api.post("/cleanup")
async def run_cleanup():
    maintenance.run_cleanup();return{"ok":True,"stats":maintenance.last_stats}

# Register auth-protected API routes
app.include_router(api)

# ===============================================================================
# STARTUP
# ===============================================================================
@app.on_event("shutdown")
async def shutdown():
    analyzer.stop();gpu_monitor.stop();maintenance.stop()
    for c in feeds.values():c.stop()
    _stop_worker()

# -- Analysis worker subprocess (process isolation) ----------------------------
_worker_proc=None
def _start_worker():
    """Launch the separate analysis worker process for GIL isolation.
    Set MEVIN_WORKERS=0 to disable and analyze in-process instead."""
    global _worker_proc
    if os.environ.get("MEVIN_WORKERS","1")=="0":
        print("  Workers:     in-process (MEVIN_WORKERS=0)")
        return
    wpath=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mevin_worker.py")
    if not os.path.exists(wpath):
        print("  Workers:     in-process (mevin_worker.py not found)")
        return
    try:
        import sys
        env=dict(os.environ,MEVIN_API=f"http://127.0.0.1:{PORT}")
        _worker_proc=subprocess.Popen([sys.executable,wpath],env=env)
        print(f"  Workers:     separate process (PID {_worker_proc.pid}) - GIL isolated")
    except Exception as e:
        print(f"  Workers:     in-process (worker launch failed: {e})")

def _stop_worker():
    global _worker_proc
    if _worker_proc:
        try:_worker_proc.terminate()
        except:pass
        _worker_proc=None

if __name__=="__main__":
    s=get_all_settings();cams=get_cameras_from_db()
    # Clear any stale worker heartbeat + transient state from a previous run
    try:
        set_setting("_worker_heartbeat","");set_setting("paused","false")
    except:pass
    print("="*60+"\n  MEVIN - real-time situational video analysis\n"+"="*60)
    print(f"  Model:       {s.get('model')}  ({s.get('max_tokens')} tokens)")
    print(f"  Mode:        {'Situational (reads intent + trend)' if s.get('situational','true')=='true' else 'Snapshot'}")
    print(f"  Analysis:    parallel per-camera, {s.get('analysis_interval')}s interval")
    print(f"  Cameras:     {len(cams)}")
    for c in cams:print(f"    - {c['name']} ({c['source']})")
    _start_worker()
    print(f"  Dashboard:   http://localhost:{PORT}")
    print(f"  API Docs:    http://localhost:{PORT}/docs")
    print("="*60+"\n")
    try:
        uvicorn.run(app,host=HOST,port=PORT,log_level="warning")
    finally:
        _stop_worker()
