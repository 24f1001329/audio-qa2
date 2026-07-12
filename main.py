import os, re, json
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
AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"

REQUIRED_KEYS = ["rows","columns","mean","std","variance","min","max","median","mode","range","allowed_values","value_range","correlation"]

class Req(BaseModel):
    audio_id: str
    audio_base64: str

PROMPT = (
    "Listen to this audio very carefully. It may be spoken in ANY language "
    "(English, Korean, Japanese, Hindi, etc.). It describes a dataset specification: "
    "number of rows, one or more column names, and per-column statistics or rules.\n"
    "Return ONLY a raw JSON object (no markdown, no ```), with EXACTLY these keys:\n"
    '"rows" (integer), "columns" (array of column-name strings), '
    '"mean", "std", "variance", "min", "max", "median", "mode", "range", '
    '"allowed_values", "value_range" (each an object mapping column name to the value '
    "stated in the audio; use {} for any not mentioned), "
    '"correlation" (array; [] if not mentioned).\n'
    "CRITICAL RULES:\n"
    "- Column names must be written EXACTLY as spoken, in the ORIGINAL language/script "
    "(e.g. if the audio says a Korean word like 온도, write 온도 in Korean characters — do NOT translate to English).\n"
    "- Every column mentioned in the audio MUST appear in the columns array.\n"
    "- The keys inside mean/std/min/max/etc. must be those same original-language column names.\n"
    "- Numbers must be plain JSON numbers exactly as spoken.\n"
    "- Include only what the audio states."
)

async def handle(req: Req):
    audio = re.sub(r"^data:audio/\w+;base64,", "", req.audio_base64)
    payload = {
        "model": "gpt-4o-audio-preview",
        "temperature": 0,
        "modalities": ["text"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    }
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(AIPIPE_URL, json=payload, headers=headers)
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        text = "{}"
    text = re.sub(r"^```(json)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(m.group(0)) if m else {}
    defaults = {"rows": 0, "columns": [], "correlation": []}
    for k in REQUIRED_KEYS:
        if k not in result:
            result[k] =