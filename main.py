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
CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"

REQUIRED_KEYS = ["rows", "columns", "mean", "std", "variance", "min", "max", "median", "mode", "range", "allowed_values", "value_range", "correlation"]

class Req(BaseModel):
    audio_id: str
    audio_base64: str

def detect_format(raw):
    if raw[:4] == b"RIFF":
        return "wav"
    if raw[:3] == b"ID3" or (len(raw) > 1 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0):
        return "mp3"
    return "wav"

PARSE_PROMPT = (
    "Below is a transcript of an audio (it may be Korean, Japanese, English, or any language). "
    "It describes a dataset specification: number of rows, one or more column names, "
    "and per-column statistics or rules.\n"
    "Return ONLY a raw JSON object (no markdown, no backticks), with EXACTLY these keys: "
    "rows (integer), columns (array of column-name strings), "
    "mean, std, variance, min, max, median, mode, range, allowed_values, value_range "
    "(each an object mapping column name to the stated value; empty object if not mentioned), "
    "correlation (array; empty array if not mentioned).\n"
    "RULES: Column names EXACTLY as in the transcript, in the ORIGINAL script "
    "(Korean stays in Hangul like 온도, never translated to English). "
    "Every column mentioned must be in the columns array. Stat object keys use those same names. "
    "Numbers as plain JSON numbers. Include only what the transcript states.\n\n"
    "Transcript:\n"
)

async def handle(req: Req):
    b64 = re.sub(r"^data:audio/[\w.+-]+;base64,", "", req.audio_base64)
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raw = b""
    fmt = detect_format(raw)
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}

    debug = {}

    async with httpx.AsyncClient(timeout=120) as client:
        t_payload = {
            "model": "gpt-4o-audio-preview",
            "temperature": 0,
            "modalities": ["text"],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                    {"type": "text", "text": "Transcribe this audio EXACTLY, word for word, in its original language and script (Korean in Hangul, Japanese in Japanese, etc.). Output only the transcription, nothing else."},
                ],
            }],
        }
        tr = await client.post(CHAT_URL, json=t_payload, headers=headers)
        debug["step1_status"] = tr.status_code
        debug["step1_raw"] = tr.text[:800]
        try:
            transcript = tr.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            transcript = ""
            debug["step1_error"] = str(e)
        debug["transcript"] = transcript

        p_payload = {
            "model": "gpt-4o",
            "temperature": 0,
            "messages": [{"role": "user", "content": PARSE_PROMPT + transcript}],
        }
        r = await client.post(CHAT_URL, json=p_payload, headers=headers)
        debug["step2_status"] = r.status_code
        debug["step2_raw"] = r.text[:800]

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        text = "{}"
        debug["step2_error"] = str(e)
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text)
    except Exception as e:
        debug["parse_error"] = str(e)
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