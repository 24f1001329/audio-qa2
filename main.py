import os, re, json, base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
GEMINI_URL = f"https://aipipe.org/gemini/v1beta/models/gemini-2.0-flash:generateContent?key={AIPIPE_TOKEN}"

REQUIRED_KEYS = ["rows", "columns", "mean", "std", "variance", "min", "max", "median", "mode", "range", "allowed_values", "value_range", "correlation"]

class Req(BaseModel):
    audio_id: str
    audio_base64: str

def detect_mime(raw):
    if raw[:4] == b"RIFF":
        return "audio/wav"
    if raw[:3] == b"ID3" or (len(raw) > 1 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0):
        return "audio/mp3"
    if raw[:4] == b"OggS":
        return "audio/ogg"
    return "audio/wav"

PROMPT = (
    "Listen to this audio very carefully, word by word. It may be spoken in ANY language "
    "(English, Korean, Japanese, Hindi, Tamil, etc.). It describes a dataset specification: "
    "number of rows, one or more column names, and per-column statistics or rules.\n"
    "Return ONLY a raw JSON object (no markdown, no backticks), with EXACTLY these keys: "
    "rows (integer), columns (array of column-name strings), "
    "mean, std, variance, min, max, median, mode, range, allowed_values, value_range "
    "(each an object mapping column name to the stated value; empty object if not mentioned), "
    "correlation (array; empty array if not mentioned).\n"
    "RULES: Column names EXACTLY as spoken, in the ORIGINAL script "
    "(Korean stays in Hangul like 온도, never translated to English). "
    "Every column mentioned must be in the columns array. Stat object keys use those same names. "
    "Numbers as plain JSON numbers. Include only what the audio states."
)

async def handle(req: Req):
    b64 = re.sub(r"^data:audio/[\w.+-]+;base64,", "", req.audio_base64)
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raw = b""
    mime = detect_mime(raw)

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime, "data": b64}},
                {"text": PROMPT},
            ]
        }]
    }

    debug = {}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(GEMINI_URL, json=payload)
    debug["status"] = r.status_code
    debug["raw"] = r.text[:800]

    try:
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        text = "{}"
        debug["error"] = str(e)

    text = text.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(m.group(0)) if m else {}

    defaults = {"rows": 0, "columns": [], "correlation": []}
    for k in REQUIRED_KEYS:
        if k not in result:
            result[k] = defaults.get(k, {})

    result["_debug"] = debug
    return result

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/")
async def root_post(req: Req):
    return await handle(req)

@app.post("/answer-audio")
async def answer_audio(req: Req):
    return await handle(req)