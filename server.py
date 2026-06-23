#
# FastAPI server — Plivo webhook + WebSocket audio stream
#
# Flow:
#   1. Plivo calls your number → POST /answer → returns XML Stream verb
#   2. Plivo opens WebSocket → WS /ws → sends start event (with streamId/callId)
#   3. Server extracts IDs, spawns an isolated bot pipeline per call
#   4. Bidirectional audio flows; bot auto-hangs up on EndFrame
#

import asyncio
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse
from google.genai import Client
from google.genai.types import GenerateContentConfig, HttpOptions
from loguru import logger
from pydantic import BaseModel
from starlette.websockets import WebSocketState

load_dotenv(override=True)

from bot import get_shared_client, prewarm_gemini, run_bot, trim_prompt_for_live


async def _prewarm_greeting_model() -> None:
    """Warm the native-audio Live model path (DNS/TLS/handshake) at startup.

    Opens and immediately closes a Live session so the first real greeting
    render connects faster and reliably finishes within the ring window.
    Non-fatal — greeting pre-render is best-effort regardless.
    """
    try:
        voice = os.getenv("GEMINI_VOICE", "Puck")
        config = _greeting_live_config(system_instruction=None, voice=voice)
        async with asyncio.timeout(15):
            async with get_shared_client().aio.live.connect(model=_greeting_model(), config=config):
                pass  # entering the context manager performs the connect handshake
        logger.info(f"Greeting model (native-audio) pre-warmed (voice={voice})")
    except Exception as e:
        logger.warning(f"Greeting model pre-warm failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the server accepts any requests.

    Pre-warms:
    - google-genai SDK (imports, object init)
    - DNS + TLS for generativelanguage.googleapis.com
    - Shared Gemini API client (reused across all calls)
    - Native-audio Live model path (so the first pre-rendered greeting isn't cold)

    Result: first call connects to Gemini Live ~2–3× faster than cold start.
    """
    logger.info("Server starting — pre-warming Gemini services …")
    await prewarm_gemini()
    await _prewarm_greeting_model()
    logger.info("All services ready. Accepting calls.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="Plivo Gemini Live Phone Bot", lifespan=lifespan)

# Per-call system_prompt store — keyed by short call_sid passed in URL
# Entries are cleaned up after the WebSocket session ends
_call_prompts: dict[str, str] = {}
_call_voices: dict[str, str] = {}
# Per-call pre-rendered greeting: (pcm_bytes, sample_rate) and the spoken text,
# keyed by call_sid. Rendered during the ring window in make_outbound_call and
# consumed (popped) in websocket_endpoint. Absence → fall back to live greeting.
_call_greetings: dict[str, tuple[bytes, int]] = {}
_call_greeting_texts: dict[str, str] = {}

# Model used to pre-render the greeting clip. MUST match the model the live
# conversation uses so the greeting voice is identical to the rest of the call.
# Honors GEMINI_LIVE_MODEL (same override bot.py uses) and falls back to the
# pipecat default native-audio model.
_GREETING_MODEL_DEFAULT = "models/gemini-2.5-flash-native-audio-preview-12-2025"


def _greeting_model() -> str:
    return os.getenv("GEMINI_LIVE_MODEL") or _GREETING_MODEL_DEFAULT


# ── Prompt lookup (mirrors ui_server.py so /call accepts instruction_id) ──────

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)


def _safe_name(name: str) -> str:
    """Sanitize a prompt filename — same rules as ui_server._safe_name."""
    name = name.removesuffix(".txt").strip()
    name = re.sub(r"[^\w\-]", "_", name)
    if not name:
        raise HTTPException(status_code=400, detail="Invalid prompt name")
    return name


# ── Greeting pre-render (mask Gemini Live cold-start) ─────────────────────────

def _extract_greeting_line(prompt: Optional[str], customer_name: Optional[str]) -> Optional[str]:
    """Pull the bot's opening line out of a system prompt for pre-rendering.

    The prompts put the greeting as the first double-quoted string under the
    ``START:`` heading, e.g.::

        START:
        Greet the user by name: "Hi {customer_name}! This is Swift. ..."

    Returns the quoted line with ``{customer_name}`` substituted, or None if no
    greeting line can be found (caller then falls back to the live greeting).
    """
    if not prompt:
        return None
    m = re.search(r"START\s*:", prompt, re.IGNORECASE)
    region = prompt[m.end():] if m else prompt
    q = re.search(r'"([^"]+)"', region)
    if not q:
        return None
    line = q.group(1).strip()
    line = line.replace("{customer_name}", customer_name or "there")
    return line or None


def _greeting_live_config(system_instruction: Optional[str], voice: str):
    """Build a LiveConnectConfig for one-shot greeting generation.

    Mirrors the pipeline's connect config (AUDIO modality, same voice,
    thinking disabled) so the rendered greeting matches the live conversation.
    """
    from google.genai.types import (
        AudioTranscriptionConfig,
        LiveConnectConfig,
        Modality,
        PrebuiltVoiceConfig,
        SpeechConfig,
        ThinkingConfig,
        VoiceConfig,
    )

    # Set fields directly on LiveConnectConfig — the generation_config wrapper
    # is deprecated in google-genai.
    config = LiveConnectConfig(
        response_modalities=[Modality.AUDIO],
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice)),
        ),
        output_audio_transcription=AudioTranscriptionConfig(),
    )
    if system_instruction:
        config.system_instruction = system_instruction
    config.thinking_config = ThinkingConfig(thinking_budget=0)
    return config


async def prerender_greeting(
    call_sid: str,
    prompt: Optional[str],
    customer_name: Optional[str],
    voice: str,
) -> None:
    """Render the greeting during the ring window using the SAME native-audio
    model the live conversation uses (so the voice is identical).

    Opens a short throwaway Gemini Live session, sends the greeting trigger,
    captures the generated audio (24kHz PCM) + its transcription, then closes.
    Stores PCM in ``_call_greetings[call_sid]`` and text in
    ``_call_greeting_texts[call_sid]``. On any failure both dicts are left clean
    so the WebSocket handler falls through to the existing live-greeting path.
    """
    try:
        # If the prompt wasn't resolved at /call time, mirror bot.py's fallback.
        if not prompt:
            try:
                prompt = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")
            except Exception:
                prompt = None
        if not prompt:
            logger.info(f"prerender_greeting: no prompt available | call_sid={call_sid}")
            return

        system_instruction = prompt.replace("{customer_name}", customer_name or "the caller")
        system_instruction = trim_prompt_for_live(system_instruction)
        fallback_text = _extract_greeting_line(prompt, customer_name)

        from google.genai.types import Content, Part

        config = _greeting_live_config(system_instruction, voice)
        pcm = bytearray()
        text_parts: list[str] = []

        # 20s ceiling so a stuck turn can never hang the background task.
        async with asyncio.timeout(20):
            async with get_shared_client().aio.live.connect(model=_greeting_model(), config=config) as session:
                await session.send_client_content(
                    turns=[Content(role="user", parts=[Part(
                        text="The call has connected. Begin the conversation per your instructions."
                    )])],
                    turn_complete=True,
                )
                async for message in session.receive():
                    sc = getattr(message, "server_content", None)
                    if not sc:
                        continue
                    if sc.model_turn and sc.model_turn.parts:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                pcm.extend(part.inline_data.data)
                    if sc.output_transcription and sc.output_transcription.text:
                        text_parts.append(sc.output_transcription.text)
                    if sc.turn_complete:
                        break

        if not pcm:
            logger.warning(f"prerender_greeting: no audio generated | call_sid={call_sid}")
            return

        # Prefer the actually-spoken transcript; fall back to the prompt's START line.
        greeting_text = ("".join(text_parts).strip() or (fallback_text or "")).strip()

        _call_greetings[call_sid] = (bytes(pcm), 24000)
        _call_greeting_texts[call_sid] = greeting_text
        logger.info(
            f"prerender_greeting: cached {len(pcm)} bytes (native-audio) | "
            f"call_sid={call_sid} voice={voice} text={greeting_text!r}"
        )
    except Exception as e:
        logger.error(f"prerender_greeting failed (call_sid={call_sid}): {e}")
        # Never leave a partial entry — guarantees a clean live-greeting fallback.
        _call_greetings.pop(call_sid, None)
        _call_greeting_texts.pop(call_sid, None)


def _discard_call(call_sid: str) -> None:
    """Drop all stored per-call state for a call that won't proceed."""
    _call_prompts.pop(call_sid, None)
    _call_voices.pop(call_sid, None)
    _call_greetings.pop(call_sid, None)
    _call_greeting_texts.pop(call_sid, None)


# ── Helpers ────────────────────────────────────────────────────────────────────

class _PlivoWebSocketProxy:
    """Thin proxy around FastAPI WebSocket.

    Pre-reads Plivo's `start` event to extract ``streamId`` / ``callId``
    before handing control to Pipecat's transport layer.  Any messages
    received *before* the start event are buffered and replayed transparently.
    """

    def __init__(self, websocket: WebSocket):
        self._ws = websocket
        self._buffer: list[dict] = []
        self.stream_id: str = "unknown"
        self.call_id: Optional[str] = None

    async def wait_for_start(self, timeout: float = 15.0) -> None:
        """Block until the Plivo `start` event arrives (or timeout)."""
        try:
            async with asyncio.timeout(timeout):
                while True:
                    raw = await self._ws.receive()
                    if raw.get("type") == "websocket.disconnect":
                        self._buffer.append(raw)
                        logger.warning("WebSocket disconnected before start event")
                        return

                    text = raw.get("text") or ""
                    if not text and raw.get("bytes"):
                        text = raw["bytes"].decode("utf-8", errors="ignore")

                    if not text:
                        self._buffer.append(raw)
                        continue

                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        self._buffer.append(raw)
                        continue

                    if data.get("event") == "start":
                        start = data.get("start", {})
                        self.stream_id = start.get("streamId", "unknown")
                        self.call_id = (
                            start.get("callId")
                            or start.get("callUUID")
                            or start.get("call_uuid")
                        )
                        logger.info(
                            f"Plivo start event | stream_id={self.stream_id} "
                            f"call_id={self.call_id}"
                        )
                        return
                    else:
                        # Buffer non-start messages so Pipecat can still process them
                        self._buffer.append(raw)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Plivo start event; continuing anyway")

    # ── WebSocket interface (used by FastAPIWebsocketClient internally) ────────

    @property
    def client_state(self):
        return self._ws.client_state

    @property
    def application_state(self):
        return self._ws.application_state

    async def receive(self) -> dict:
        """Return buffered messages first, then forward to the real socket."""
        if self._buffer:
            return self._buffer.pop(0)
        return await self._ws.receive()

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def send_text(self, data: str) -> None:
        await self._ws.send_text(data)

    async def close(self, code: int = 1000) -> None:
        try:
            await self._ws.close(code)
        except Exception:
            pass


# ── Routes ─────────────────────────────────────────────────────────────────────

class CallRequest(BaseModel):
    to: str                              # E.164 number to dial, e.g. "+919876543210"
    from_: Optional[str] = None          # Override PLIVO_FROM_NUMBER from .env
    system_prompt: Optional[str] = None  # Per-call prompt override (full text)
    instruction_id: Optional[str] = None # Filename (without .txt) inside prompts/; loaded if system_prompt is not set
    customer_name: Optional[str] = None  # Customer name injected into system prompt
    lead_id: Optional[str] = None    # Lead UUID passed back in the webhook callback
    callback_url: Optional[str] = None   # URL to POST extracted lead data when call ends
    voice: Optional[str] = None          # Gemini voice: Puck | Aoede | Charon | Fenrir | Kore | Leda | Orus | Zephyr

@app.post("/call")
async def make_outbound_call(req: CallRequest):
    """Trigger an outbound call via Plivo REST API.

    Plivo dials `to`, and when answered it hits our /answer webhook which
    starts the Gemini Live audio stream — same flow as an inbound call.

    Example::
        curl -X POST http://localhost:8090/call \\
             -H 'Content-Type: application/json' \\
             -d '{"to": "+919876543210", "lead_id": "550e8400-e29b-41d4-a716-446655440000", "customer_name": "Rahul", "callback_url": "https://your-service.com/lead"}'
    """
    auth_id = os.getenv("PLIVO_AUTH_ID")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN")
    from_number = req.from_ or os.getenv("PLIVO_FROM_NUMBER")
    ngrok_host = os.getenv("PUBLIC_HOST") or os.getenv("NGROK_HOST")

    if not auth_id or not auth_token:
        raise HTTPException(status_code=500, detail="PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN not set")
    if not from_number:
        raise HTTPException(status_code=500, detail="PLIVO_FROM_NUMBER not set in .env")
    if not ngrok_host:
        raise HTTPException(status_code=500, detail="PUBLIC_HOST (or NGROK_HOST) not set in .env")

    from urllib.parse import quote
    # Store system_prompt and voice server-side to avoid URL length limits
    call_sid = uuid.uuid4().hex[:16]

    # Resolve the prompt — order: explicit text > instruction_id file > SYSTEM_PROMPT env
    # (bot.py applies the system_prompt.txt fallback if everything below is empty.)
    effective_prompt: Optional[str] = None
    prompt_source = "default"
    if req.system_prompt:
        effective_prompt = req.system_prompt
        prompt_source = "system_prompt"
    elif req.instruction_id:
        path = PROMPTS_DIR / f"{_safe_name(req.instruction_id)}.txt"
        if not path.exists():
            raise HTTPException(
                status_code=404, detail=f"Instruction '{req.instruction_id}' not found"
            )
        effective_prompt = path.read_text(encoding="utf-8")
        prompt_source = f"instruction_id={req.instruction_id}"
    elif os.getenv("SYSTEM_PROMPT"):
        effective_prompt = os.getenv("SYSTEM_PROMPT")
        prompt_source = "env"

    logger.info(
        f"/call prompt source: {prompt_source} "
        f"({len(effective_prompt or '')} chars) instruction_id={req.instruction_id!r}"
    )

    if effective_prompt:
        _call_prompts[call_sid] = effective_prompt
    if req.voice:
        _call_voices[call_sid] = req.voice

    greeting_voice = req.voice or os.getenv("GEMINI_VOICE", "Puck")

    params = []
    if req.customer_name: params.append(f"customer_name={quote(req.customer_name, safe='')}")
    if req.lead_id is not None: params.append(f"lead_id={req.lead_id}")
    if req.callback_url:  params.append(f"callback_url={quote(req.callback_url, safe='')}")
    params.append(f"call_sid={call_sid}")
    # to_number is passed so the bot can include it in the callback payload
    params.append(f"to_number={quote(req.to, safe='')}")
    use_wss = os.getenv("USE_WSS", "true").lower() == "true"
    answer_scheme = "https" if use_wss else "http"
    # PUBLIC_PORT: set when running on a raw IP (e.g. EC2 elastic IP) without a reverse proxy.
    # Leave unset when behind ngrok or a load balancer (standard 443/80 ports).
    public_port = os.getenv("PUBLIC_PORT", "")
    host_with_port = f"{ngrok_host}:{public_port}" if public_port else ngrok_host
    answer_url = f"{answer_scheme}://{host_with_port}/answer" + ("?" + "&".join(params) if params else "")
    # Plivo accepts E.164 with or without leading '+'; strip it to avoid format mismatches
    from_clean = from_number.lstrip("+")
    to_clean = req.to.lstrip("+")
    payload = {
        "from": from_clean,
        "to": to_clean,
        "answer_url": answer_url,
        "answer_method": "POST",
    }
    endpoint = f"https://api.plivo.com/v1/Account/{auth_id}/Call/"

    # Best-effort greeting pre-render in the background. It races the ring: the
    # greeting is instant when the render finishes before pickup, and falls back
    # to the live greeting otherwise. The dial below stays synchronous so the call
    # goes out immediately and Plivo errors surface in the /call response.
    asyncio.create_task(
        prerender_greeting(call_sid, effective_prompt, req.customer_name, greeting_voice)
    )

    logger.info(f"Calling Plivo API | from={from_clean} to={to_clean} answer_url={answer_url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            endpoint,
            json=payload,
            auth=aiohttp.BasicAuth(auth_id, auth_token),
        ) as response:
            body = await response.json()
            if response.status not in (200, 201, 202):
                logger.error(f"Plivo outbound call failed: {response.status} {body}")
                _discard_call(call_sid)
                raise HTTPException(status_code=response.status, detail=body)

    logger.info(f"Outbound call initiated | to={req.to} answer_url={answer_url}")
    return JSONResponse({"status": "calling", "to": req.to, "answer_url": answer_url})


@app.post("/answer")
async def answer_call(request: Request):
    """Plivo webhook — answers the call and streams audio to this server."""
    # PUBLIC_HOST (or legacy NGROK_HOST) takes priority; falls back to the request Host header
    host = os.getenv("PUBLIC_HOST") or os.getenv("NGROK_HOST") or request.headers.get("host", "yourdomain.com")
    ws_scheme = "wss" if os.getenv("USE_WSS", "true").lower() == "true" else "ws"
    public_port = os.getenv("PUBLIC_PORT", "")
    host = f"{host}:{public_port}" if public_port else host

    from urllib.parse import quote
    qp = request.query_params
    params = []
    if qp.get("customer_name"): params.append(f"customer_name={quote(qp['customer_name'], safe='')}")
    if qp.get("lead_id"):   params.append(f"lead_id={qp['lead_id']}")
    if qp.get("callback_url"):  params.append(f"callback_url={quote(qp['callback_url'], safe='')}")
    if qp.get("call_sid"):      params.append(f"call_sid={qp['call_sid']}")
    if qp.get("to_number"):     params.append(f"to_number={quote(qp['to_number'], safe='')}")
    ws_url = f"{ws_scheme}://{host}/ws" + ("?" + "&".join(params) if params else "")

    # & in query strings is invalid XML — must be escaped as &amp; inside XML elements
    ws_url_xml = ws_url.replace("&", "&amp;")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream streamTimeout="86400"
            keepCallAlive="true"
            bidirectional="true"
            contentType="audio/x-mulaw;rate=8000"
            maxDuration="3600">
        {ws_url_xml}
    </Stream>
</Response>"""

    logger.info(f"Answering call — streaming to {ws_url}")
    return PlainTextResponse(xml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    customer_name: Optional[str] = None,
    lead_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    call_sid: Optional[str] = None,
    to_number: Optional[str] = None,
):
    """WebSocket endpoint — one pipeline per connected call."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # Retrieve and consume per-call overrides
    system_prompt = _call_prompts.pop(call_sid, None) if call_sid else None
    system_prompt = system_prompt or os.getenv("SYSTEM_PROMPT")
    voice = _call_voices.pop(call_sid, None) if call_sid else None

    # Consume the pre-rendered greeting clip (if it finished during the ring).
    greeting = _call_greetings.pop(call_sid, None) if call_sid else None
    greeting_text = _call_greeting_texts.pop(call_sid, None) if call_sid else None
    greeting_pcm, greeting_rate = greeting if greeting else (None, None)

    # Step 1: Extract Plivo metadata from start event
    proxy = _PlivoWebSocketProxy(websocket)
    await proxy.wait_for_start()

    if proxy.stream_id == "unknown":
        logger.warning("No valid stream_id — call may not work correctly")

    # Step 2: Run the bot (isolated pipeline for this call)
    try:
        await run_bot(
            websocket=proxy,
            stream_id=proxy.stream_id,
            call_id=proxy.call_id,
            system_prompt=system_prompt,
            customer_name=customer_name,
            lead_id=lead_id,
            callback_url=callback_url,
            to_number=to_number,
            voice=voice,
            greeting_pcm=greeting_pcm,
            greeting_rate=greeting_rate,
            greeting_text=greeting_text,
        )
    except Exception as e:
        logger.error(f"Bot error for stream_id={proxy.stream_id}: {e}", exc_info=True)
    finally:
        # Close if not already closed
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass
        logger.info(f"WebSocket closed | stream_id={proxy.stream_id}")


# ── AI Prompt Generator ────────────────────────────────────────────────────────

_gen_client: Optional[Client] = None


def _get_gen_client() -> Client:
    global _gen_client
    if _gen_client is None:
        _gen_client = Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=HttpOptions(api_version="v1beta"),
        )
    return _gen_client


class GeneratePromptRequest(BaseModel):
    use_case: str
    company_name: str = "our company"
    bot_name: str = "AI Assistant"
    language: str = "auto"          # auto | english | hindi | hinglish
    tone: str = "friendly"          # friendly | professional | casual | assertive
    fields_to_collect: str = ""
    qualification_criteria: str = ""
    additional_context: str = ""


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

PLATFORM FACTS (your generated prompt must account for all of these):
- Telephony: Plivo outbound call, 8 kHz u-law audio — quality is often noisy or muffled.
- AI engine: Google Gemini Live (real-time, streaming voice — NOT text chat).
- One runtime template variable is available: {{customer_name}}
  This is substituted with the real caller's name before every call. Use it naturally.
- The bot speaks first (greeter role). It must never wait silently.
- Response latency target: < 1.5 s — keep outputs SHORT.
- VAD (voice activity detection) cuts the bot off the moment the user speaks — so long monologues get interrupted.

BOT SPECIFICATION:
Company   : {req.company_name}
Bot name  : {req.bot_name}
Purpose   : {req.use_case}
Language  : {lang_rule}
Tone      : {tone_guide}
Collect   : {fields}
Qualify by: {qualification}{extra}

REQUIRED SECTIONS — include every one, in this order:

1. ROLE (2-3 sentences) — who the bot is, which company, the single goal.
2. AUDIO NOTE (1 sentence) — if caller is unclear, ask to repeat once. Never guess.
3. CONVERSATION STYLE (bullet list) — CRITICAL: every response <= 20 words, one sentence, never longer; ask exactly one question per turn; never list options unless asked; never admit to being AI unless asked directly; apply language and tone rules.
4. KNOWN INFORMATION — list what is already known so bot never asks again. Always include: "- Caller's name: {{customer_name}}"
5. INFORMATION TO COLLECT — numbered steps, one specific data point per step, ordered by priority.
6. INTENT CLASSIFICATION — define four intents with one-line trigger each: INTERESTED / EXPLORING / NOT_INTERESTED / CALLBACK
7. CONVERSATION FLOW — subsections: START / IF INTERESTED / IF EXPLORING / IF NOT INTERESTED / IF CALLBACK / IF CALLER ASKS A QUESTION. Each has 1-3 example lines the bot says, <= 20 words each. START must use {{customer_name}}.
8. QUALIFICATION — define what a HIGH QUALITY outcome looks like. Include exact script line when caller qualifies.
9. NEXT STEPS — what bot says and confirms when caller agrees to proceed.
10. RULES (bullet list, >= 5 rules) — hard constraints the bot must never break.

OUTPUT FORMAT:
- Plain text only. No markdown, no code fences.
- Section headers in UPPER CASE.
- Example scripts in quotation marks.
- NO preamble. Start directly with the ROLE section.
- NO closing note or explanation after last section.
- Total length: 400-700 words.
"""


@app.post("/generate-prompt")
async def generate_prompt(req: GeneratePromptRequest):
    """Use Gemini to generate a production-ready phone bot system prompt.

    Example::
        curl -X POST http://localhost:8090/generate-prompt \\
             -H 'Content-Type: application/json' \\
             -d '{
               "use_case": "Solar panel lead qualification",
               "company_name": "Swift Solar",
               "bot_name": "Swift",
               "language": "hinglish",
               "tone": "friendly",
               "fields_to_collect": "location, property type, electricity bill, roof type, ownership"
             }'
    """
    if not req.use_case.strip():
        raise HTTPException(status_code=400, detail="use_case is required")

    meta = _build_meta_prompt(req)
    logger.info(f"Generating prompt | use_case={req.use_case!r} lang={req.language} tone={req.tone}")

    try:
        client = _get_gen_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8090"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)