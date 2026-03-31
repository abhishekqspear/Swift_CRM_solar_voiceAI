#
# UI Server — Bot Management Dashboard (port 8081)
#
# Serves a single-page management UI and provides:
#   - CRUD API for system prompt files (stored in prompts/)
#   - Call proxy: reads prompt file, forwards to main bot server /call
#
# Run:
#   python ui_server.py
#   → http://localhost:8081
#

import asyncio
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from google.genai import Client
from google.genai.types import GenerateContentConfig, HttpOptions
from loguru import logger
from pydantic import BaseModel

load_dotenv(override=True)

# ── Config ──────────────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

BOT_SERVER = os.getenv("BOT_SERVER_URL", "http://localhost:8000")

app = FastAPI(title="Swift Solar Bot Dashboard")

# In-memory lead store — last 50 webhook results
_leads: deque = deque(maxlen=50)

# Shared Gemini client for prompt generation
_gen_client: Optional[Client] = None


def _get_gen_client() -> Client:
    global _gen_client
    if _gen_client is None:
        _gen_client = Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=HttpOptions(api_version="v1beta"),
        )
    return _gen_client


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    """Sanitize a prompt filename.

    Strips the .txt suffix (if any), then replaces every character that is not
    alphanumeric, a dash, or an underscore with '_'.  Raises HTTP 400 if the
    result is empty.
    """
    name = name.removesuffix(".txt").strip()
    name = re.sub(r"[^\w\-]", "_", name)
    if not name:
        raise HTTPException(status_code=400, detail="Invalid prompt name")
    return name


# ── Static ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "ui" / "index.html")


# ── Prompt CRUD ───────────────────────────────────────────────────────────────

@app.get("/api/prompts")
async def list_prompts():
    """List all prompt files sorted by last-modified (newest first)."""
    files = sorted(
        PROMPTS_DIR.glob("*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "name": f.stem,
            "size": f.stat().st_size,
            "updated_at": int(f.stat().st_mtime),
        }
        for f in files
    ]


@app.get("/api/prompts/{name}")
async def get_prompt(name: str):
    """Return the content of a single prompt file."""
    path = PROMPTS_DIR / f"{_safe_name(name)}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"name": path.stem, "content": path.read_text(encoding="utf-8")}


class PromptBody(BaseModel):
    name: str
    content: str


@app.post("/api/prompts", status_code=201)
async def create_prompt(body: PromptBody):
    """Create a new prompt file.  Returns 409 if the name already exists."""
    name = _safe_name(body.name)
    path = PROMPTS_DIR / f"{name}.txt"
    if path.exists():
        raise HTTPException(status_code=409, detail="A prompt with that name already exists")
    path.write_text(body.content, encoding="utf-8")
    logger.info(f"Created prompt: {name}")
    return {"name": name}


@app.put("/api/prompts/{name}")
async def update_prompt(name: str, body: PromptBody):
    """Overwrite an existing prompt file.  Supports rename (old name in URL, new name in body)."""
    old_path = PROMPTS_DIR / f"{_safe_name(name)}.txt"
    if not old_path.exists():
        raise HTTPException(status_code=404, detail="Prompt not found")

    new_name = _safe_name(body.name)
    new_path = PROMPTS_DIR / f"{new_name}.txt"

    # Prevent overwriting a *different* existing file
    if new_path != old_path and new_path.exists():
        raise HTTPException(status_code=409, detail="Target name already exists")

    new_path.write_text(body.content, encoding="utf-8")
    if new_path != old_path:
        old_path.unlink()
        logger.info(f"Renamed prompt: {_safe_name(name)} → {new_name}")
    else:
        logger.info(f"Updated prompt: {new_name}")

    return {"name": new_name}


@app.delete("/api/prompts/{name}", status_code=204)
async def delete_prompt(name: str):
    """Delete a prompt file."""
    path = PROMPTS_DIR / f"{_safe_name(name)}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prompt not found")
    path.unlink()
    logger.info(f"Deleted prompt: {_safe_name(name)}")


# ── Call proxy ────────────────────────────────────────────────────────────────

class CallRequest(BaseModel):
    to: str
    customer_name: Optional[str] = None
    customer_id: Optional[int] = None
    callback_url: Optional[str] = None
    instruction_id: Optional[str] = None  # filename (without .txt) of the system prompt to use


@app.post("/api/call")
async def make_call(req: CallRequest):
    """Resolve the instruction file (if given) then forward to the bot server's /call endpoint."""
    prompt_text: Optional[str] = None
    if req.instruction_id:
        path = PROMPTS_DIR / f"{_safe_name(req.instruction_id)}.txt"
        if not path.exists():
            raise HTTPException(
                status_code=404, detail=f"Instruction '{req.instruction_id}' not found"
            )
        prompt_text = path.read_text(encoding="utf-8")

    # Build payload — omit None values so the bot server uses its own defaults
    payload: dict = {"to": req.to}
    if req.customer_name:
        payload["customer_name"] = req.customer_name
    if req.customer_id is not None:
        payload["customer_id"] = req.customer_id
    if req.callback_url:
        payload["callback_url"] = req.callback_url
    if prompt_text:
        payload["system_prompt"] = prompt_text

    logger.info(
        f"Proxying call → {BOT_SERVER}/call | to={req.to} "
        f"customer={req.customer_name} instruction_id={req.instruction_id or '(server default)'}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BOT_SERVER}/call",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            body = await resp.json()
            if resp.status not in (200, 201, 202):
                logger.error(f"Bot server /call failed: {resp.status} {body}")
                raise HTTPException(status_code=resp.status, detail=body)
            return body


# ── Lead webhook receiver ─────────────────────────────────────────────────────

@app.post("/api/leads", status_code=200)
async def receive_lead(request: Request):
    """Receive lead data POSTed by the bot after a call ends.

    Set callback_url to http://<this-host>:8081/api/leads when making a call
    and the result will appear in the Leads tab of the dashboard.
    """
    try:
        data: Any = await request.json()
    except Exception:
        data = {}
    entry = {"received_at": int(time.time()), **data}
    _leads.appendleft(entry)
    logger.info(
        f"Lead received | customer_id={data.get('customer_id')} "
        f"phone={data.get('phone_number')}"
    )
    return {"status": "ok"}


@app.get("/api/leads")
async def list_leads():
    """Return stored lead results (newest first, max 50)."""
    return list(_leads)


# ── AI Prompt Generator ───────────────────────────────────────────────────────

class GeneratePromptRequest(BaseModel):
    use_case: str                          # required — what the bot should do
    company_name: str = "our company"
    bot_name: str = "AI Assistant"
    language: str = "auto"                 # auto | english | hindi | hinglish
    tone: str = "friendly"                 # friendly | professional | casual | assertive
    fields_to_collect: str = ""            # what information to gather from the caller
    qualification_criteria: str = ""       # what makes a good outcome / qualified lead
    additional_context: str = ""           # any extra instructions or constraints


def _build_meta_prompt(req: GeneratePromptRequest) -> str:
    lang_rule = {
        "auto":      "Detect and mirror the caller's language. Support Hindi, English, and Hinglish (Hindi-English mix) seamlessly.",
        "english":   "Respond in English throughout.",
        "hindi":     "Respond in Hindi throughout, using Devanagari-friendly phrasing.",
        "hinglish":  "Respond in Hinglish — a natural, conversational mix of Hindi and English as spoken in urban India.",
    }.get(req.language.lower(), "Detect and mirror the caller's language.")

    tone_guide = {
        "friendly":     "Warm, approachable, encouraging — like a helpful friend, not a salesperson.",
        "professional": "Polished, respectful, confident — like a senior business executive.",
        "casual":       "Relaxed, informal, conversational — use contractions and everyday language.",
        "assertive":    "Clear, direct, persuasive — guide the conversation purposefully without being pushy.",
    }.get(req.tone.lower(), "Warm and approachable.")

    fields = req.fields_to_collect.strip() or "Relevant details based on the use case"
    qualification = req.qualification_criteria.strip() or "Determine based on the user's intent and responses"
    extra = f"\nADDITIONAL CONSTRAINTS:\n{req.additional_context.strip()}" if req.additional_context.strip() else ""

    return f"""You are a world-class conversation designer specialising in AI voice bots for phone calls. \
Your output will be used verbatim as a system prompt for a real production phone bot — it must be complete, \
precise, and immediately usable without any editing.

━━━ PLATFORM FACTS (your generated prompt must account for all of these) ━━━
• Telephony: Plivo outbound call, 8 kHz µ-law audio — quality is often noisy or muffled.
• AI engine: Google Gemini Live (real-time, streaming voice — NOT text chat).
• One runtime template variable is available: {{customer_name}}
  → This is substituted with the real caller's name before every call. Use it naturally.
• The bot speaks first (greeter role). It must never wait silently.
• Response latency target: < 1.5 s — keep outputs SHORT.
• VAD (voice activity detection) cuts the bot off the moment the user speaks — so long monologues get interrupted.

━━━ BOT SPECIFICATION ━━━
Company   : {req.company_name}
Bot name  : {req.bot_name}
Purpose   : {req.use_case}
Language  : {lang_rule}
Tone      : {tone_guide}
Collect   : {fields}
Qualify by: {qualification}{extra}

━━━ REQUIRED SECTIONS — include every one, in this order ━━━

1. ROLE (2–3 sentences)
   Who the bot is, which company it represents, and the single goal of this call.

2. AUDIO NOTE (1 sentence)
   If the caller is unclear, ask them to repeat once politely. Never guess.

3. CONVERSATION STYLE (bullet list)
   • CRITICAL: every response ≤ 20 words. One sentence. Never longer.
   • Ask exactly one question per turn.
   • Never list options unless the user asks.
   • Never mention you are an AI unless asked directly.
   • Language rule (from spec above).
   • Tone rule (from spec above).

4. KNOWN INFORMATION
   List what is already known so the bot never asks for it again.
   Always include: "- Caller's name: {{customer_name}}"

5. INFORMATION TO COLLECT
   Numbered steps. Each step = one specific data point to gather.
   Order them from most to least important.

6. INTENT CLASSIFICATION
   Define exactly four intents with a one-line trigger description each:
   INTERESTED | EXPLORING | NOT_INTERESTED | CALLBACK

7. CONVERSATION FLOW
   Subsections: START / IF INTERESTED / IF EXPLORING / IF NOT INTERESTED / IF CALLBACK / IF CALLER ASKS A QUESTION
   Each subsection has 1–3 concrete example lines the bot says, ≤ 20 words each.
   START must use {{customer_name}}.

8. QUALIFICATION
   Define what a HIGH QUALITY outcome looks like.
   Include the exact script line the bot says when the caller qualifies.

9. NEXT STEPS
   What the bot says and confirms once the caller agrees to proceed.

10. RULES (bullet list, ≥ 5 rules)
    Hard constraints the bot must never break.

━━━ OUTPUT FORMAT ━━━
• Plain text only — absolutely no markdown, no ```, no bullet symbols other than "-" or "•".
• Section headers in UPPER CASE followed by a colon or blank line.
• Example scripts use quotation marks.
• NO preamble ("Here is your prompt:") — start directly with the ROLE section.
• NO closing note or explanation after the last section.
• Total length: 400–700 words. Concise but complete.
"""


@app.post("/api/generate-prompt")
async def generate_prompt(req: GeneratePromptRequest):
    """Use Gemini to generate a production-ready phone bot system prompt."""
    if not req.use_case.strip():
        raise HTTPException(status_code=400, detail="use_case is required")

    meta = _build_meta_prompt(req)
    logger.info(f"Generating prompt | use_case={req.use_case!r} lang={req.language} tone={req.tone}")

    try:
        client = _get_gen_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=meta,
            config=GenerateContentConfig(
                temperature=0.75,
                max_output_tokens=2048,
            ),
        )
        text = response.text.strip()
        logger.info(f"Prompt generated | {len(text)} chars")
        return {"prompt": text}
    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("UI_HOST", "0.0.0.0")
    port = int(os.getenv("UI_PORT", "8081"))
    logger.info(f"Starting UI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
