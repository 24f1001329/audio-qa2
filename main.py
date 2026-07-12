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

REQUIRED_KEYS = ["rows", "columns", "mean", "std", "variance", "min", "max", "median", "mode", "range", "allowed_values", "value_range", "correlation"]

class Req(BaseModel):
    audio_id: str
    audio_base64: str

PROMPT = (
    "Listen to this audio very carefully. It may be spoken in ANY language "
    "(English, Korean, Japanese, Hindi, etc.). It describes a dataset specification: "
    "number of rows, one or more column names, and per-column statistics or rules. "
    "Return ONLY a raw JSON object (no markdown, no backticks), with EXACTLY these keys: "
    "rows (integer), columns (array of column-name strings), "
    "mean, std, variance, min, max, median, mode, range, "
    "allowed_values, value_range (each an object mapping column name to the value "
    "stated in the audio; use an empty object for any not mentioned), "
    "correlation (array; empty array if not mentioned). "
    "CRITICAL RULES: "
    "1) Column names must be written EXACTLY as spoken, in the ORIGINAL language and script. "
    "For example if the audio says a Korean word, write it in Korean characters. Do NOT translate to English. "
    "2) Every column mentioned in the audio MUST appear in the columns array. "
    "3) The keys inside mean, std, min, max and the other stat objects must be those same original-language column names. "
    "4) Numbers must be plain JSON numbers exactly as spoken. "
    "5) Include only what the audio states."
)

async def handle(req: Req):
    audio = re.sub(r"^data:audio/\w+;base64,", "", req.audio_base64)
    payload = {
        "model": "gpt-4o-audio-preview",
        "temperature": 0,
        "modalities": ["text"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(AIPIPE_URL, json=payload, headers=headers)
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        text = "{}"
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            result = json.loads(m.group(0))
        else:
            result = {}
    defaults = {"rows": 0, "columns": [], "correlation": []}
    for k in REQUIRED_KEYS:
        if k not in result:
            result[k] = defaults.get(k, {})
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