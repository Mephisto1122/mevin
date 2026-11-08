"""
Mevin core - pure analysis logic shared by the web process (mevin.py) and the
analysis worker (mevin_worker.py).

Everything here is dependency-light (stdlib only) so it can be unit-tested
without cameras, a GPU, Ollama, or FastAPI.
"""
import html
import os
import re
from collections import deque
from urllib.parse import quote

# ---- defaults -------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.environ.get("MEVIN_OLLAMA_URL", "http://localhost:11434")

# ---- text cleanup -------------------------------------------------------------
PREAMBLES = [
    "An image showing ", "An image of ", "The image shows ", "The scene shows ",
    "In this image, ", "In this scene, ", "I can see ", "Here is ", "Here's ",
    "This appears ", "This image shows ", "Based on the image, ",
    "The camera shows ", "The image depicts ", "In the image, ",
    "Content: ", "Analyze the Image: ", "A wide shot of ", "A shot of ",
    "An analysis of the scene", "An analysis of ", "Analysis: ",
    "Here is my analysis", "Here is the analysis", "Scene analysis:",
    "Security analysis:", "Scene description:", "Summary: ",
    "An outdoor ", "A view of ", "Looking at ", "We can see ",
    "What is happening? ", "Observation: ",
    "The updated image:.", "The updated image:", "Updated image:",
    "The image:.", "Image analysis:", "Scene:",
    "A description of the scene:.", "A description of the scene:",
    "A description of the scene", "Description of the scene:",
    "Description:", "Here is a description",
]
THINK_MARKERS = [
    "Thinking Process", "thinking process", "Analyze the Request", "Analyze the Image",
    "Goal:", "Output Format:", "Required Content:", "Step ", "**Analyze", "**Examine",
    "**Setting:", "**Subject", "**Action", "**Object", "Answer the prompt",
]
_THINK_DROP = ["analyze", "thinking", "goal:", "format:", "step ", "required",
               "output:", "process", "answer the prompt"]
# Token prefixes that mean the model started "thinking out loud" instead of answering
STREAM_THINK_PREFIXES = ["analyze the", "thinking process", "goal:", "output format",
                         "step 1", "**analyze"]


def clean(t, prompt=""):
    """Turn raw VLM output into one tidy situational sentence (max 3)."""
    t = (t or "").strip()
    # Strip prompt echo - models sometimes repeat the prompt
    if prompt:
        t = t.replace(prompt, "").strip()
        for frag in prompt.split("."):
            frag = frag.strip()
            if len(frag) > 10:
                t = t.replace('"' + frag + '"', "").replace("'" + frag + "'", "")
                t = t.replace(frag, "").strip()
    # Strip thinking/analysis blocks
    if any(m in t for m in THINK_MARKERS):
        parts = [p.strip() for p in t.split("\n\n") if p.strip()]
        keep = []
        for p in reversed(parts):
            pc = re.sub(r"\*\*[^*]+\*\*:?\s*", "", p).strip()
            if pc and not any(m in pc.lower() for m in _THINK_DROP):
                keep.insert(0, pc)
                if len(keep) >= 2:
                    break
        if keep:
            t = " ".join(keep)
        else:
            last = t.split(".")
            t = ". ".join(last[-3:]).strip() if len(last) > 3 else t
    # Strip markdown
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\d+\.\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\*\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\n{2,}", ". ", t)
    t = re.sub(r"\n", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    # Strip preambles - loop until no more matches
    changed = True
    while changed:
        changed = False
        for p in PREAMBLES:
            if t.lower().startswith(p.lower()):
                t = t[len(p):].strip()
                changed = True
    t = re.sub(r"^[\"':.\s]+", "", t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    sents = re.split(r"(?<=[.!?])\s+", t)
    if len(sents) > 3:
        t = " ".join(sents[:3])
    return t.strip()


def is_thinking_start(text):
    low = (text or "").lower()
    return any(m in low for m in STREAM_THINK_PREFIXES)


# ---- alert keywords -----------------------------------------------------------
NEG_RE = re.compile(
    r"\b(no|not|don't|didn't|doesn't|without|never|absence|lack|free of|no sign of|"
    r"isn't|aren't|wasn't|weren't|nobody|none)\b", re.I)
_CLAUSE_BREAK = re.compile(r"[.;!?,]|\bbut\b|\bhowever\b|\bwhile\b", re.I)
# Common inflections so "attack" still catches "attacked", but whole words only:
# "armed" no longer fires on "unarmed", "fire" not on "fireplace", "drag" not on "dragon".
_SUFFIX = r"(?:s|es|ed|d|ing)?"


def _keyword_re(k):
    return re.compile(r"(?<![\w-])" + re.escape(k.lower()) + _SUFFIX + r"(?![\w-])")


def match_keywords(caption, keywords):
    """Return the keywords that genuinely appear in the caption.

    Whole-word matching (with plural/past/-ing forms) and negation-aware:
    "no weapon visible" does not trigger "weapon", but "nobody is hurt, a man
    is fighting" still triggers "fight" because the negation is in another clause.
    """
    low = (caption or "").lower()
    triggered = []
    for k in keywords or []:
        if not k or not str(k).strip():
            continue
        for m in _keyword_re(str(k).strip()).finditer(low):
            window = low[max(0, m.start() - 50):m.start()]
            # Only look at negations within the same clause as the keyword
            breaks = list(_CLAUSE_BREAK.finditer(window))
            if breaks:
                window = window[breaks[-1].end():]
            if not NEG_RE.search(window):
                triggered.append(k)
                break
    return triggered


def danger_score(triggered, no_change, motion_pct):
    """0..1 danger estimate used for the per-camera trend line."""
    danger = 0.02 if no_change else 0.1
    if triggered:
        danger = 0.5 + min(len(triggered) * 0.12, 0.4)
    danger += min(max(motion_pct, 0) / 100, 1) * 0.1
    return round(min(1.0, danger), 2)


# ---- scene memory -------------------------------------------------------------
SMALL_MODEL_HINTS = ["moondream", "moon", "1.6b", "0.5b", "tiny", "nano"]


def is_small_model(model):
    low = (model or "").lower()
    return any(m in low for m in SMALL_MODEL_HINTS)


class SceneMemory:
    """A living, situational understanding of one camera.
    Tracks what's normal, what's been happening, and where danger is trending."""

    def __init__(self):
        self.baseline = ""                     # learned "normal" for this camera
        self.history = deque(maxlen=6)         # recent (time_str, caption, danger)
        self.danger_trail = deque(maxlen=20)   # danger scores over time
        self.calm_count = 0                    # consecutive calm frames
        self.last_situation = ""

    def build_context(self, base_prompt, model=""):
        """Compose a situational prompt: baseline + recent narrative + the ask.
        Small models get a compact version - they lose focus on long prompts."""
        if is_small_model(model):
            if self.baseline:
                return (f"Normally: {self.baseline[:80]}. Now - {base_prompt} "
                        "Note if anything changed or looks dangerous.")
            return base_prompt
        parts = []
        if self.baseline:
            parts.append(f"NORMAL for this camera: {self.baseline}")
        if self.history:
            recent = "\n".join(f"  [{t}] {c}" for t, c, _ in self.history)
            parts.append(f"WHAT JUST HAPPENED:\n{recent}")
        if self.baseline or self.history:
            parts.append(
                base_prompt +
                " Read the SITUATION, not just objects: what are people doing, what is their"
                " intent, how does this differ from normal, and is it escalating, stable, or"
                " calming. Call out crime or danger the moment you see it. One sentence.")
        else:
            parts.append(base_prompt)
        return "\n\n".join(parts)

    def trend(self):
        """Danger trajectory: rising / falling / stable."""
        if len(self.danger_trail) < 4:
            return "stable"
        r = list(self.danger_trail)
        h = len(r) // 2
        first = sum(r[:h]) / max(1, h)
        last = sum(r[h:]) / max(1, len(r) - h)
        if last > first + 0.15:
            return "rising"
        if last < first - 0.15:
            return "falling"
        return "stable"

    def update(self, caption, danger, time_str):
        self.history.append((time_str, caption, danger))
        self.danger_trail.append(danger)
        self.last_situation = caption
        # Learn baseline from calm, stable moments
        if danger < 0.1:
            self.calm_count += 1
            if self.calm_count >= 3 and (not self.baseline or self.calm_count % 20 == 0):
                self.baseline = caption
        else:
            self.calm_count = 0


# ---- misc helpers -------------------------------------------------------------
def ollama_base(url):
    """Normalise whatever the user typed into a base URL (no /api/... suffix)."""
    url = (url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/v1/chat/completions", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def inject_rtsp_credentials(uri, user, password):
    """rtsp://host/... -> rtsp://user:pass@host/... with credentials URL-encoded,
    so passwords containing @ : / # etc. don't break the URL."""
    if not user or "@" in uri.split("://", 1)[-1].split("/", 1)[0]:
        return uri
    creds = quote(user, safe="") + ":" + quote(password or "", safe="")
    return re.sub(r"^(rtsps?://)", lambda m: m.group(1) + creds + "@", uri, count=1)


def safe_join(base, rel):
    """Resolve rel inside base; return None if it escapes base (path traversal)."""
    base_real = os.path.realpath(base)
    target = os.path.realpath(os.path.join(base_real, rel.lstrip("/\\")))
    if target != base_real and not target.startswith(base_real + os.sep):
        return None
    return target


def tg_escape(text):
    """Escape text for Telegram parse_mode=HTML (model output can contain < > &)."""
    return html.escape(text or "", quote=False)
