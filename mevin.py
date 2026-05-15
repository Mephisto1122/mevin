"""
Mevin | Real-time AI video analysis.
pip install fastapi uvicorn opencv-python requests numpy python-multipart
python mevin.py -> http://localhost:5555
API docs -> http://localhost:5555/docs
"""
import asyncio,base64,json,os,pathlib,socket,sqlite3,subprocess,threading,time
from collections import deque
from datetime import datetime,timedelta
from queue import Empty,Queue
from typing import Optional
import cv2,numpy as np,requests as req
import uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,JSONResponse,StreamingResponse,Response

PORT=5555;HOST="0.0.0.0";DB_PATH="monitor.db";SNAPSHOT_DIR="snapshots"
STREAM_FPS=12;THUMB_WIDTH=280;JPEG_QUALITY=75
DEFAULTS={
    "model":"gemma3:4b","ollama_url":"http://localhost:11434",
    "prompt":"What is happening? Count people, describe actions, flag danger. Two sentences max.",
    "alert_keywords":json.dumps(["weapon","knife","gun","fight","fighting","attack","punch","kick","aggressive","threatening","suspicious","intruder","stranger","trespassing","break-in","forced","smash","fallen","falling","unconscious","injured","bleeding","fire","smoke","flame","running","shouting","screaming","panic","unattended","abandoned","mask","covered face","hoodie","loitering","hiding","crawling","climbing","vandalism","theft","stealing","robbery"]),
    "max_tokens":"200","inference_size":"448","analysis_interval":"5",
    "telegram_token":"","telegram_chat_id":"","telegram_on_alert":"true","telegram_on_all":"false",
    "telegram_quiet_start":"","telegram_quiet_end":"","telegram_min_interval":"30",
    "motion_sensitivity":"0.3","motion_enabled":"true",
    "snapshot_on_every":"false","snapshot_quality":"90",
    "feed_show_stable":"true","feed_max_items":"200","setup_done":"false",
    "retain_days":"30","retain_max_obs":"5000","retain_max_snap_mb":"2000",
    "cleanup_interval_min":"60","thumb_retain_days":"7",
}

app=FastAPI(title="Mevin",description="Real-time AI video analysis",docs_url="/docs")
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
        _db_conn.execute("PRAGMA busy_timeout=10000")
    return _db_conn
def db_exec(sql,params=(),fetch=False,fetchone=False):
    with _db_lock:
        c=get_shared_db()
        r=c.execute(sql,params)
        if fetchone:return r.fetchone()
        if fetch:return r.fetchall()
        c.commit();return r.rowcount
def db_exec_script(sql):
    with _db_lock:
        c=get_shared_db();c.executescript(sql)
def init_db():
    db_exec_script("""
        CREATE TABLE IF NOT EXISTS cameras(id TEXT PRIMARY KEY,name TEXT,source TEXT,enabled INTEGER DEFAULT 1,sort_order INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,camera_id TEXT,camera_name TEXT,timestamp TEXT,text TEXT,is_alert INTEGER DEFAULT 0,alert_keywords TEXT DEFAULT '[]',snapshot_path TEXT,thumb_b64 TEXT,pinned INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(timestamp DESC);
    """)
    if db_exec("SELECT COUNT(*) FROM cameras",fetchone=True)[0]==0:
        db_exec("INSERT INTO cameras VALUES('cam0','Camera 1','0',1,0)")
def get_setting(k):
    r=db_exec("SELECT value FROM settings WHERE key=?",(k,),fetchone=True)
    return r["value"] if r else DEFAULTS.get(k,"")
def set_setting(k,v):db_exec("REPLACE INTO settings(key,value)VALUES(?,?)",(k,str(v)))
def get_all_settings():
    rows=db_exec("SELECT key,value FROM settings",fetch=True)
    s=dict(DEFAULTS)
    for r in rows:s[r["key"]]=r["value"]
    return s
def get_cameras_from_db():
    rows=db_exec("SELECT * FROM cameras WHERE enabled=1 ORDER BY sort_order",fetch=True)
    return [dict(r) for r in rows]
def save_observation(obs):
    db_exec("INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)",
        (obs["id"],obs["camera_id"],obs["camera_name"],obs["timestamp"],obs["text"],obs.get("is_alert",0),json.dumps(obs.get("alert_keywords",[])),obs.get("snapshot_path",""),obs.get("thumb_b64",""),0))
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

# ===============================================================================
# CAMERA FEED
# ===============================================================================
class CameraFeed:
    def __init__(s,cfg):
        s.id=cfg["id"];s.name=cfg["name"];s.source=cfg["source"];s.frame=None;s.lock=threading.Lock();s.running=True;s.connected=False
        s.is_file=not str(s.source).isdigit() and not str(s.source).startswith(("rtsp://","http://","https://"))
        threading.Thread(target=s._run,daemon=True).start()
    def _run(s):
        while s.running:
            src=int(s.source)if str(s.source).isdigit()else s.source
            cap=cv2.VideoCapture(src)
            if not cap.isOpened():s.connected=False;time.sleep(5);continue
            s.connected=True;cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            fps=cap.get(cv2.CAP_PROP_FPS)if s.is_file else 0
            if fps<=0 or fps>120:fps=25
            delay=1.0/fps if s.is_file else 0
            print(f"[{s.id}] Connected: {s.name}"+( f" (file {fps:.0f}fps)"if s.is_file else""))
            while s.running:
                ok,f=cap.read()
                if not ok:
                    if s.is_file:cap.set(cv2.CAP_PROP_POS_FRAMES,0);continue
                    s.connected=False;break
                with s.lock:s.frame=f
                if s.is_file:time.sleep(delay)
            cap.release()
    def get_frame(s):
        with s.lock:return s.frame.copy()if s.frame is not None else None
    def thumbnail(s,frame):
        h,w=frame.shape[:2];sc=THUMB_WIDTH/w;t=cv2.resize(frame,(THUMB_WIDTH,int(h*sc)),interpolation=cv2.INTER_AREA)
        _,jpg=cv2.imencode(".jpg",t,[cv2.IMWRITE_JPEG_QUALITY,70]);return base64.b64encode(jpg.tobytes()).decode()
    def mjpeg_gen(s):
        interval=1.0/STREAM_FPS
        while s.running:
            f=s.get_frame()
            if f is None:b=np.zeros((360,640,3),dtype=np.uint8);_,jpg=cv2.imencode(".jpg",b)
            else:_,jpg=cv2.imencode(".jpg",f,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
            yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+jpg.tobytes()+b"\r\n"
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
# VLM ANALYZER
# ===============================================================================
class VLMAnalyzer:
    def __init__(s):
        s.cameras={};s.sse_queues=[];s.sse_lock=threading.Lock();s.running=True
        s.prev_caption={};s.prev_gray={};s.no_change_count={};s.focus_cam=None;s.frames_per_cam=3;s.cur_frame=0
        threading.Thread(target=s._loop,daemon=True).start()
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
                      "What is happening? ","Observation: "]:
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
        isz=sint(st.get("inference_size","448"),448);h,w=frame.shape[:2];sc=isz/max(h,w)
        sm=cv2.resize(frame,(int(w*sc),int(h*sc)),interpolation=cv2.INTER_AREA)if sc<1 else frame
        _,jpg=cv2.imencode(".jpg",sm,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
        b64=base64.b64encode(jpg.tobytes()).decode();thumb=cam.thumbnail(frame)
        ts=datetime.now();iid=f"{cam.id}_{int(ts.timestamp()*1000)}"
        bp=st.get("prompt",DEFAULTS["prompt"]);prev=s.prev_caption.get(cam.id,"")
        ctx=f"{bp}\nPrevious: \"{prev}\"\nWhat changed? If nothing, say \"No change.\""if prev else bp
        print(f"[{cam.name}] motion={mp:.1f}% analyzing...")
        s._broadcast({"type":"stream_start","camera_id":cam.id,"item_id":iid,"camera_name":cam.name,"time":ts.strftime("%H:%M:%S"),"thumb":thumb,"motion":round(mp,1)})
        caption=""
        model=st.get("model","gemma3:4b")
        base_url=st.get("ollama_url",DEFAULTS["ollama_url"]).replace("/api/generate","").replace("/api/chat","").rstrip("/")
        opts={"temperature":0.2,"num_predict":sint(st.get("max_tokens","200"),200)}
        try:
            # Use OpenAI-compatible endpoint (stable across Ollama versions)
            r=req.post(f"{base_url}/v1/chat/completions",json={
                "model":model,
                "messages":[{"role":"user","content":[
                    {"type":"text","text":ctx},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]}],
                "stream":True,
                "max_tokens":sint(st.get("max_tokens","200"),200),
                "temperature":0.2,
            },headers={"Content-Type":"application/json"},stream=True,timeout=120)
            if r.status_code>=400:
                try:err=r.json().get("error",{});emsg=err.get("message","") if isinstance(err,dict) else str(err)
                except:emsg=r.text[:300]
                print(f"  [API] /v1/chat/completions error ({r.status_code}): {emsg}")
                # Fallback to native /api/chat
                r=req.post(f"{base_url}/api/chat",json={"model":model,"messages":[{"role":"user","content":ctx,"images":[b64]}],"stream":True,"keep_alive":"30m","options":opts},stream=True,timeout=120)
                if r.status_code>=400:
                    try:err2=r.json().get("error","")
                    except:err2=r.text[:200]
                    print(f"  [API] /api/chat fallback error ({r.status_code}): {err2}")
            r.raise_for_status()
            thinking="";content_started=False
            for line in r.iter_lines():
                if not s.running:break
                if not line:continue
                line_str=line.decode("utf-8",errors="ignore") if isinstance(line,bytes) else line
                # Skip SSE prefix
                if line_str.startswith("data: "):line_str=line_str[6:]
                if line_str.strip()=="[DONE]":break
                try:
                    chunk=json.loads(line_str)
                    # OpenAI format: choices[0].delta.content
                    tk=""
                    if "choices" in chunk:
                        delta=chunk["choices"][0].get("delta",{})
                        tk=delta.get("content","") or ""
                    else:
                        # Native Ollama format
                        tk=chunk.get("response","") or chunk.get("message",{}).get("content","") or ""
                    th=chunk.get("message",{}).get("thinking","") or chunk.get("thinking","") or ""
                    if th:thinking+=th;continue  # collect thinking silently
                    if tk:
                        # Skip if this looks like thinking leaked into content
                        if not content_started:
                            test=(caption+tk).lower()
                            if any(m in test for m in ['analyze the','thinking process','goal:','output format','required output','step 1','**analyze']):
                                caption+=tk;continue  # collect but don't broadcast
                            content_started=True
                            caption=tk  # reset, discard any preamble
                        else:
                            caption+=tk
                        s._broadcast({"type":"stream_token","item_id":iid,"text":s._clean(caption,bp)})
                except:continue
            # If model only produced thinking, use it
            if not caption.strip() and thinking.strip():caption=thinking
        except Exception as e:caption=f"Error: {e}"
        caption=s._clean(caption.strip(),bp)or"(no response)"
        no_chg=caption.lower().strip().startswith("no change")
        if no_chg:s.no_change_count[cam.id]=s.no_change_count.get(cam.id,0)+1
        else:s.no_change_count[cam.id]=0;s.prev_caption[cam.id]=caption
        if cam.id not in s.prev_caption:s.prev_caption[cam.id]=caption
        kws=json.loads(st.get("alert_keywords","[]"));triggered=[k for k in kws if k.lower()in caption.lower()];is_alert=len(triggered)>0
        snap=""
        if is_alert or st.get("snapshot_on_every","false")=="true":
            dd=os.path.join(SNAPSHOT_DIR,ts.strftime("%Y-%m-%d"));os.makedirs(dd,exist_ok=True)
            sn=f"{cam.id}_{ts.strftime('%H%M%S')}.jpg";snap=os.path.join(dd,sn);cv2.imwrite(snap,frame,[cv2.IMWRITE_JPEG_QUALITY,sint(st.get("snapshot_quality","90"),90)])
        if not no_chg or is_alert:save_observation({"id":iid,"camera_id":cam.id,"camera_name":cam.name,"timestamp":ts.isoformat(),"text":caption,"is_alert":int(is_alert),"alert_keywords":triggered,"snapshot_path":snap,"thumb_b64":thumb})
        s._broadcast({"type":"feed_item","item_id":iid,"camera_id":cam.id,"camera_name":cam.name,"time":ts.strftime("%H:%M:%S"),"text":caption,"thumb":thumb,"is_alert":is_alert,"alert_keywords":triggered,"no_change":no_chg,"motion":round(mp,1),"snapshot":snap.replace("\\","/")if snap else""})
        if is_alert and st.get("telegram_on_alert","true")=="true":threading.Thread(target=send_telegram,args=(f"ALERT {cam.name} {ts.strftime('%H:%M:%S')}\n{caption}\nKeywords: {', '.join(triggered)}",snap),daemon=True).start()
        elif st.get("telegram_on_all","false")=="true"and not no_chg:threading.Thread(target=send_telegram,args=(f"{cam.name} {ts.strftime('%H:%M:%S')}\n{caption}",),daemon=True).start()
    def _loop(s):
        s.last_analyzed={}   # cam_id -> timestamp
        s.analysis_times={}  # cam_id -> last duration in seconds
        s.load_warned=False
        while s.running:
            cams=list(s.cameras.values())
            if not cams:time.sleep(1);continue

            # Pick camera
            if s.focus_cam and s.focus_cam in s.cameras:
                cam=s.cameras[s.focus_cam]
            else:
                # Round-robin by least-recently-analyzed
                connected=[c for c in cams if c.connected]
                if not connected:time.sleep(2);continue
                cam=min(connected,key=lambda c:s.last_analyzed.get(c.id,0))

            if cam.connected:
                t0=time.time()
                s._analyze(cam)
                dur=time.time()-t0
                s.last_analyzed[cam.id]=time.time()
                s.analysis_times[cam.id]=dur

                # Broadcast GPU stats + camera rotation status
                gpu=gpu_monitor.to_dict()
                n_cams=len([c for c in cams if c.connected])
                cycle_time=sum(s.analysis_times.values()) if s.analysis_times else 0
                status_data={
                    "type":"gpu","data":gpu,
                    "rotation":{
                        "current":cam.name,
                        "total_cameras":n_cams,
                        "cycle_time":round(cycle_time,1),
                        "per_camera":{cid:round(dt,1) for cid,dt in s.analysis_times.items()},
                        "focus":s.focus_cam,
                    }
                }

                # Load warnings
                overloaded=False
                warns=[]
                if gpu.get("available"):
                    if gpu["vram_pct"]>90:warns.append(f"VRAM at {gpu['vram_pct']}%");overloaded=True
                    if gpu["temp"]>82:warns.append(f"GPU temp {gpu['temp']}degC");overloaded=True
                if n_cams>1 and cycle_time>n_cams*15:
                    warns.append(f"Full cycle takes {cycle_time:.0f}s for {n_cams} cameras")
                    overloaded=True
                if dur>30:warns.append(f"{cam.name} took {dur:.0f}s");overloaded=True

                if overloaded and not s.load_warned:
                    status_data["load_warning"]="; ".join(warns)
                    print(f"[LOAD WARNING] {'; '.join(warns)}")
                    s.load_warned=True
                elif not overloaded:s.load_warned=False

                s._broadcast(status_data)
            else:time.sleep(2)
            st=get_all_settings();time.sleep(max(0.2,sfloat(st.get("analysis_interval","0.5"),0.5)))
    def subscribe(s):
        q=Queue(maxsize=100)
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
    """Add one camera without disrupting others."""
    global feeds
    feeds[cam_dict["id"]]=CameraFeed(cam_dict)
    analyzer.set_cameras(feeds)

def remove_single_camera(cid):
    """Remove one camera without disrupting others."""
    global feeds
    if cid in feeds:feeds[cid].stop();del feeds[cid]
    analyzer.set_cameras(feeds)

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
async def video_feed(cam_id:str):
    cam=feeds.get(cam_id)
    if not cam:return Response(status_code=404)
    def resized():
        interval=1.0/STREAM_FPS
        while cam.running:
            f=cam.get_frame()
            if f is None:
                b=np.zeros((int(MJPEG_WIDTH*9/16),MJPEG_WIDTH,3),dtype=np.uint8)
                _,jpg=cv2.imencode(".jpg",b)
            else:
                h,w=f.shape[:2]
                if w>MJPEG_WIDTH:
                    sc=MJPEG_WIDTH/w;f=cv2.resize(f,(MJPEG_WIDTH,int(h*sc)),interpolation=cv2.INTER_AREA)
                _,jpg=cv2.imencode(".jpg",f,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
            yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+jpg.tobytes()+b"\r\n"
            time.sleep(interval)
    return StreamingResponse(resized(),media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/events")
async def sse_events():
    q=analyzer.subscribe()
    async def gen():
        loop=asyncio.get_event_loop()
        try:
            while True:
                try:
                    msg=await asyncio.wait_for(loop.run_in_executor(None,lambda:q.get(timeout=2)),timeout=3)
                    yield f"data:{json.dumps(msg)}\n\n"
                except(Empty,asyncio.TimeoutError):
                    yield ":\n\n"
        except asyncio.CancelledError:analyzer.unsubscribe(q)
        except GeneratorExit:analyzer.unsubscribe(q)
    return StreamingResponse(gen(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

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
    d=await req.json();cid=f"cam{int(time.time()*1000)}"
    db_exec("INSERT INTO cameras(id,name,source)VALUES(?,?,?)",(cid,d["name"],d["source"]))
    add_single_camera({"id":cid,"name":d["name"],"source":d["source"]})
    return{"ok":True,"id":cid}

@api.put("/cameras/{cid}")
async def update_camera(cid:str,req:Request):
    d=await req.json()
    db_exec("UPDATE cameras SET name=?,source=? WHERE id=?",(d["name"],d["source"],cid))
    remove_single_camera(cid)
    add_single_camera({"id":cid,"name":d["name"],"source":d["source"]})
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
    d=await req.json();analyzer.set_focus(d.get("camera_id"));return{"ok":True}

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
async def recent_feed(limit:int=50):
    """Load recent observations from DB for page reload."""
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

if __name__=="__main__":
    s=get_all_settings();cams=get_cameras_from_db()
    print("="*60+"\n  MEVIN (FastAPI)\n"+"="*60)
    print(f"  Model:     {s.get('model')}\n  Cameras:   {len(cams)}")
    for c in cams:print(f"    - {c['name']} ({c['source']})")
    print(f"  Dashboard: http://localhost:{PORT}")
    print(f"  API Docs:  http://localhost:{PORT}/docs")
    print(f"  Retention: {s.get('retain_days')}d / max {s.get('retain_max_obs')} obs / {s.get('retain_max_snap_mb')}MB snaps")
    print("="*60+"\n")
    uvicorn.run(app,host=HOST,port=PORT,log_level="warning")
