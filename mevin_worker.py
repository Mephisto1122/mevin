"""
Mevin analysis worker - runs in a SEPARATE PROCESS for GIL isolation.

Launched automatically by mevin.py (set MEVIN_WORKERS=0 to disable).
It fetches frames from the web process over localhost, runs motion detection +
the VLM, and posts results back. Because it's a separate process, the heavy
VLM token-parsing never competes with the web server's GIL - the dashboard
stays responsive no matter how busy analysis is.

State (cameras, settings, pause, focus) is read from the shared SQLite DB.
A heartbeat is written every few seconds; if this process dies, the web
process detects the stale heartbeat and resumes analysis in-thread.
"""
import base64,json,os,re,sqlite3,threading,time
from collections import deque
from datetime import datetime
import cv2,numpy as np,requests as req

API=os.environ.get("MEVIN_API","http://127.0.0.1:5555")
DB_PATH=os.environ.get("MEVIN_DB","monitor.db")
GPU_SLOTS=int(os.environ.get("MEVIN_GPU_SLOTS","1"))

_NEG_RE=re.compile(r"\b(no|not|don't|didn't|doesn't|without|never|absence|lack|free of|no sign of|isn't|aren't|wasn't|weren't)\b",re.I)

# ---- DB access (read settings, write observations + heartbeat) ----------------
_conn=sqlite3.connect(DB_PATH,check_same_thread=False,timeout=30)
_conn.row_factory=sqlite3.Row
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA busy_timeout=30000")
_dblock=threading.Lock()

def settings():
    with _dblock:
        rows=_conn.execute("SELECT key,value FROM settings").fetchall()
    return {r["key"]:r["value"] for r in rows}

def set_kv(k,v):
    with _dblock:
        _conn.execute("INSERT INTO settings(key,value)VALUES(?,?)ON CONFLICT(key)DO UPDATE SET value=?",(k,str(v),str(v)))
        _conn.commit()

def sint(v,d=0):
    try:return int(float(v))
    except:return d
def sfloat(v,d=0.0):
    try:return float(v)
    except:return d

# ---- text cleanup (mirrors mevin.py _clean) -----------------------------------
_PREAMBLES=["An image showing ","An image of ","The image shows ","The scene shows ",
    "In this image, ","In this scene, ","I can see ","Here is ","Here's ",
    "This appears ","This image shows ","Based on the image, ","The camera shows ",
    "The image depicts ","In the image, ","Content: ","Analyze the Image: ",
    "A wide shot of ","A shot of ","An analysis of the scene","An analysis of ",
    "Analysis: ","Here is my analysis","Here is the analysis","Scene analysis:",
    "Security analysis:","Scene description:","Summary: ","An outdoor ","A view of ",
    "Looking at ","We can see ","What is happening? ","Observation: ",
    "The updated image:.","The updated image:","Updated image:","The image:.",
    "Image analysis:","Scene:","A description of the scene:.","A description of the scene:",
    "A description of the scene","Description of the scene:","Description:","Here is a description"]

def clean(t,prompt=""):
    t=(t or "").strip()
    if prompt:
        t=t.replace(prompt,"").strip()
        for frag in prompt.split('.'):
            frag=frag.strip()
            if len(frag)>10:
                t=t.replace(frag,"").replace('"'+frag+'"',"").replace("'"+frag+"'",'').strip()
    markers=['Thinking Process','thinking process','Analyze the Request','Analyze the Image',
             'Goal:','Output Format:','Required Content:','Step ','**Analyze','**Examine',
             '**Setting:','**Subject','**Action','**Object','Answer the prompt']
    if any(m in t for m in markers):
        parts=[p.strip() for p in t.split('\n\n') if p.strip()];keep=[]
        for p in reversed(parts):
            pc=re.sub(r'\*\*[^*]+\*\*:?\s*','',p).strip()
            if pc and not any(m.lower() in pc.lower() for m in ['analyze','thinking','goal:','format:','step ','required','output:','process','answer the prompt']):
                keep.insert(0,pc)
                if len(keep)>=2:break
        if keep:t=' '.join(keep)
    t=re.sub(r'\*\*([^*]+)\*\*',r'\1',t);t=re.sub(r'\*([^*]+)\*',r'\1',t)
    t=re.sub(r'^#+\s*','',t,flags=re.MULTILINE);t=re.sub(r'^\d+\.\s*','',t,flags=re.MULTILINE)
    t=re.sub(r'\n{2,}','. ',t);t=re.sub(r'\n',' ',t);t=re.sub(r'\s{2,}',' ',t)
    changed=True
    while changed:
        changed=False
        for p in _PREAMBLES:
            if t.lower().startswith(p.lower()):t=t[len(p):].strip();changed=True
    t=re.sub(r'^["\':.\s]+','',t)
    if t and t[0].islower():t=t[0].upper()+t[1:]
    sents=re.split(r'(?<=[.!?])\s+',t)
    if len(sents)>3:t=' '.join(sents[:3])
    return t.strip()

# ---- scene memory (per camera) ------------------------------------------------
class SceneMemory:
    def __init__(s):
        s.baseline="";s.history=deque(maxlen=6);s.danger_trail=deque(maxlen=20);s.calm=0
    def build_context(s,bp,model=""):
        small=any(m in model.lower() for m in["moondream","moon","1.6b","0.5b","tiny","nano"])
        if small:
            if s.baseline:return f"Normally: {s.baseline[:80]}. Now - {bp} Note if anything changed or looks dangerous."
            return bp
        parts=[]
        if s.baseline:parts.append(f"NORMAL for this camera: {s.baseline}")
        if s.history:parts.append("WHAT JUST HAPPENED:\n"+"\n".join(f"  [{t}] {c}" for t,c,_ in s.history))
        if s.baseline or s.history:
            parts.append(bp+" Read the SITUATION, not just objects: what are people doing, what is"
                " their intent, how does this differ from normal, and is it escalating, stable, or"
                " calming. Call out crime or danger the moment you see it. One sentence.")
        else:parts.append(bp)
        return "\n\n".join(parts)
    def trend(s):
        if len(s.danger_trail)<4:return"stable"
        r=list(s.danger_trail);h=len(r)//2
        first=sum(r[:h])/max(1,h);last=sum(r[h:])/max(1,len(r)-h)
        if last>first+0.15:return"rising"
        if last<first-0.15:return"falling"
        return"stable"
    def update(s,cap,danger,tm):
        s.history.append((tm,cap,danger));s.danger_trail.append(danger)
        if danger<0.1:
            s.calm+=1
            if s.calm>=3 and(not s.baseline or s.calm%20==0):s.baseline=cap
        else:s.calm=0

# ---- comms with web process ---------------------------------------------------
def fetch_frame(cid):
    """Get a high-quality frame from the web process (it owns the cameras)."""
    try:
        r=req.get(f"{API}/internal/frame-hq/{cid}",timeout=10)
        if r.status_code!=200:return None
        arr=np.frombuffer(r.content,np.uint8)
        return cv2.imdecode(arr,cv2.IMREAD_COLOR)
    except Exception:return None

def get_cameras():
    try:return req.get(f"{API}/internal/cameras",timeout=10).json()
    except Exception:return []

def emit(ev):
    try:req.post(f"{API}/internal/event",json=ev,timeout=10)
    except Exception:pass

# ---- per-camera analysis state ------------------------------------------------
prev_caption={};prev_gray={};no_change={};memory={};situation={};zone_ts={}
vlm_slots=threading.Semaphore(GPU_SLOTS)

def motion(cid,frame):
    sm=cv2.resize(frame,(320,180),interpolation=cv2.INTER_AREA)
    g=cv2.GaussianBlur(cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY),(21,21),0)
    prev=prev_gray.get(cid);prev_gray[cid]=g
    if prev is None:return 100.0
    d=cv2.absdiff(prev,g);_,th=cv2.threshold(d,25,255,cv2.THRESH_BINARY)
    return (np.count_nonzero(th)/(th.shape[0]*th.shape[1]))*100

def thumb_of(frame):
    h,w=frame.shape[:2];sc=280/w
    t=cv2.resize(frame,(280,int(h*sc)),interpolation=cv2.INTER_AREA)
    _,j=cv2.imencode(".jpg",t,[cv2.IMWRITE_JPEG_QUALITY,70]);return base64.b64encode(j.tobytes()).decode()

def mem(cid):
    if cid not in memory:memory[cid]=SceneMemory()
    return memory[cid]

def analyze(cam):
    cid=cam["id"];name=cam["name"]
    frame=fetch_frame(cid)
    if frame is None:return
    st=settings();mp=motion(cid,frame);ms=sfloat(st.get("motion_sensitivity","0.3"),0.3)
    if st.get("motion_enabled","true")=="true":
        nc=no_change.get(cid,0)
        if mp<ms and 0<nc<5:no_change[cid]=nc+1;return
    bp=st.get("prompt","People count, actions, danger. One sentence.")
    model=st.get("model","gemma3:4b")
    situational=st.get("situational","true")=="true"
    m=mem(cid)
    ctx=m.build_context(bp,model) if situational else (bp if not prev_caption.get(cid) else f"{bp}\nPrevious: \"{prev_caption.get(cid)}\"")
    caption="";base_url=st.get("ollama_url","http://localhost:11434").replace("/api/generate","").replace("/api/chat","").rstrip("/")
    last_push=0
    vlm_slots.acquire()  # wait for the GPU - may take seconds if other cameras are queued
    try:
        # CRITICAL for real-time: re-fetch the freshest frame NOW that the GPU is ours,
        # so the VLM analyzes "this instant", not a frame from before the queue wait.
        fresh=fetch_frame(cid)
        if fresh is not None:frame=fresh;mp=motion(cid,frame)
        ts=datetime.now();iid=f"{cid}_{int(ts.timestamp()*1000)}"
        thumb=thumb_of(frame)
        h,w=frame.shape[:2];isz=sint(st.get("inference_size","768"),768);sc=isz/max(h,w)
        sm=cv2.resize(frame,(int(w*sc),int(h*sc)),interpolation=cv2.INTER_AREA) if sc<1 else frame
        _,jpg=cv2.imencode(".jpg",sm,[cv2.IMWRITE_JPEG_QUALITY,92]);b64=base64.b64encode(jpg.tobytes()).decode()
        print(f"[{name}] motion={mp:.1f}% analyzing (live)...")
        emit({"type":"stream_start","camera_id":cid,"item_id":iid,"camera_name":name,"time":ts.strftime("%H:%M:%S"),"thumb":thumb,"motion":round(mp,1)})
        r=req.post(f"{base_url}/v1/chat/completions",json={
            "model":model,"messages":[{"role":"user","content":[
                {"type":"text","text":ctx},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
            "stream":True,"max_tokens":sint(st.get("max_tokens","50"),50),"temperature":0.2,"keep_alive":"30m",
        },headers={"Content-Type":"application/json"},stream=True,timeout=120)
        if r.status_code>=400:
            try:err=r.json().get("error",{});emsg=err.get("message","") if isinstance(err,dict) else str(err)
            except:emsg=r.text[:300]
            if any(x in emsg.lower() for x in["not found","no such model","try pulling"]):
                hint=f"Model '{model}' is not installed. Run: ollama pull {model}"
                emit({"type":"stream_token","item_id":iid,"text":hint});caption=hint;raise RuntimeError(hint)
            r=req.post(f"{base_url}/api/chat",json={"model":model,"messages":[{"role":"user","content":ctx,"images":[b64]}],"stream":True,"keep_alive":"30m","options":{"temperature":0.2,"num_predict":sint(st.get("max_tokens","50"),50)}},stream=True,timeout=120)
        r.raise_for_status()
        thinking="";started=False
        for line in r.iter_lines():
            if not line:continue
            ls=line.decode("utf-8",errors="ignore") if isinstance(line,bytes) else line
            if ls.startswith("data: "):ls=ls[6:]
            if ls.strip()=="[DONE]":break
            try:
                ch=json.loads(ls);tk=""
                if "choices" in ch:tk=ch["choices"][0].get("delta",{}).get("content","") or ""
                else:tk=ch.get("response","") or ch.get("message",{}).get("content","") or ""
                th=ch.get("message",{}).get("thinking","") or ch.get("thinking","") or ""
                if th:thinking+=th;continue
                if tk:
                    if not started:
                        if any(x in (caption+tk).lower() for x in["analyze the","thinking process","goal:","output format","step 1","**analyze"]):
                            caption+=tk;continue
                        started=True;caption=tk
                    else:caption+=tk
                    # throttle token pushes to ~3/sec (less localhost chatter than per-token)
                    if time.time()-last_push>0.3:
                        last_push=time.time();emit({"type":"stream_token","item_id":iid,"text":clean(caption,bp)})
            except:continue
        if not caption.strip() and thinking.strip():caption=thinking
    except RuntimeError:pass
    except Exception as e:caption=f"Error: {e}"
    finally:vlm_slots.release()
    caption=clean(caption.strip(),bp) or "(no response)"
    no_chg=caption.lower().strip().startswith("no change")
    if no_chg:no_change[cid]=no_change.get(cid,0)+1
    else:no_change[cid]=0;prev_caption[cid]=caption
    if cid not in prev_caption:prev_caption[cid]=caption
    kws=json.loads(st.get("alert_keywords","[]"));cap_low=caption.lower();triggered=[]
    for k in kws:
        kl=k.lower()
        if kl not in cap_low:continue
        pos=0;found=False
        while True:
            idx=cap_low.find(kl,pos)
            if idx<0:break
            if not _NEG_RE.search(cap_low[max(0,idx-50):idx]):found=True;break
            pos=idx+1
        if found:triggered.append(k)
    is_alert=len(triggered)>0
    danger=0.02 if no_chg else 0.1
    if is_alert:danger=0.5+min(len(triggered)*0.12,0.4)
    danger=min(1.0,danger+min(mp/100,1)*0.1)
    trend="stable"
    if situational and not no_chg:m.update(caption,danger,ts.strftime("%H:%M:%S"));trend=m.trend()
    if not no_chg:situation[cid]={"text":caption,"danger":danger,"trend":trend,"ts":time.time(),"zone":cam.get("zone","Main"),"name":name}
    # snapshot on alert (worker writes its own snapshot file)
    snap=""
    if is_alert or st.get("snapshot_on_every","false")=="true":
        try:
            dd=os.path.join("snapshots",ts.strftime("%Y-%m-%d"));os.makedirs(dd,exist_ok=True)
            snap=os.path.join(dd,f"{cid}_{ts.strftime('%H%M%S')}.jpg")
            cv2.imwrite(snap,frame,[cv2.IMWRITE_JPEG_QUALITY,sint(st.get("snapshot_quality","90"),90)])
        except Exception:snap=""
    emit({"type":"feed_item","item_id":iid,"camera_id":cid,"camera_name":name,"time":ts.strftime("%H:%M:%S"),
        "iso":ts.isoformat(),"text":caption,"thumb":thumb,"is_alert":is_alert,"alert_keywords":triggered,
        "no_change":no_chg,"motion":round(mp,1),"snapshot":snap.replace("\\","/") if snap else"",
        "trend":trend,"danger":round(danger,2),"_save":(not no_chg or is_alert)})

# ---- zone synthesis (big picture) ---------------------------------------------
def synthesize_zones():
    st=settings()
    if st.get("zone_synthesis","true")!="true":return
    zones={}
    for cid,sit in list(situation.items()):
        zones.setdefault(sit.get("zone","Main") or "Main",[]).append((sit.get("name",cid),sit))
    for zone,cams in zones.items():
        if len(cams)<2:continue
        newest=max(c[1].get("ts",0) for c in cams)
        if newest<=zone_ts.get(zone,0):continue
        zone_ts[zone]=newest
        reports="\n".join(f"  {n}: {s.get('text','')}" for n,s in cams)
        maxd=max((s.get("danger",0) for _,s in cams),default=0)
        prompt=(f"These cameras all watch the same area ({zone}). Each reports what it sees:\n{reports}\n\n"
                "In ONE sentence, describe the overall situation across this area. "
                "If a person or event appears to move between cameras, say so. Focus on the big picture.")
        model=st.get("model","gemma3:4b");base_url=st.get("ollama_url","http://localhost:11434").replace("/api/generate","").replace("/api/chat","").rstrip("/")
        vlm_slots.acquire()
        try:
            r=req.post(f"{base_url}/v1/chat/completions",json={"model":model,"messages":[{"role":"user","content":prompt}],"stream":False,"max_tokens":60,"temperature":0.3,"keep_alive":"30m"},timeout=60)
            if r.status_code<400:
                txt=(r.json().get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
                if txt:emit({"type":"zone_summary","zone":zone,"text":txt,"time":datetime.now().strftime("%H:%M:%S"),"danger":round(maxd,2),"cameras":[n for n,_ in cams]})
        except Exception as e:print(f"[zone] {e}")
        finally:vlm_slots.release()

# ---- main loop ----------------------------------------------------------------
def heartbeat_loop():
    while True:
        set_kv("_worker_heartbeat",time.time());time.sleep(3)

def zone_loop():
    while True:
        time.sleep(6)
        if settings().get("paused","false")=="true":continue
        try:synthesize_zones()
        except Exception as e:print(f"[zone] {e}")

def wait_for_web():
    for _ in range(60):
        try:
            if req.get(f"{API}/internal/cameras",timeout=3).status_code==200:return True
        except:pass
        time.sleep(1)
    return False

def main():
    print(f"[worker] starting, web API = {API}")
    if not wait_for_web():
        print("[worker] web process not reachable, exiting");return
    print("[worker] connected. Analysis running in separate process (GIL isolated).")
    threading.Thread(target=heartbeat_loop,daemon=True).start()
    threading.Thread(target=zone_loop,daemon=True).start()
    workers={}  # cam_id -> Thread
    def cam_loop(cid):
        while True:
            st=settings()
            if st.get("paused","false")=="true":time.sleep(0.5);continue
            cams={c["id"]:c for c in get_cameras()}
            cam=cams.get(cid)
            if not cam or not cam.get("connected"):return
            focus=st.get("focus_cam","")
            if focus and focus!=cid:time.sleep(1);continue
            start=time.time()
            try:analyze(cam)
            except Exception as e:print(f"[{cid}] {e}")
            # Cadence = max(interval, time the analysis took). When the GPU is the
            # bottleneck we go back-to-back on fresh frames (no wasted idle); when it's
            # free we hold the interval so we don't re-analyze the same instant.
            interval=max(0.2,sfloat(settings().get("analysis_interval","2"),2))
            time.sleep(max(0.0,interval-(time.time()-start)))
    while True:
        try:
            cams=get_cameras()
            active=[c for c in cams if c.get("connected")]
            st=settings();focus=st.get("focus_cam","")
            if focus:active=[c for c in cams if c["id"]==focus] or active
            for c in active:
                w=workers.get(c["id"])
                if w is None or not w.is_alive():
                    t=threading.Thread(target=cam_loop,args=(c["id"],),daemon=True);workers[c["id"]]=t;t.start()
        except Exception as e:print(f"[worker] loop: {e}")
        time.sleep(1)

if __name__=="__main__":
    main()
