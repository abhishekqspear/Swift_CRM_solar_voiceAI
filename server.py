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
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from pydantic import BaseModel
from starlette.websockets import WebSocketState

load_dotenv(override=True)

from bot import prewarm_gemini, run_bot


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the server accepts any requests.

    Pre-warms:
    - google-genai SDK (imports, object init)
    - DNS + TLS for generativelanguage.googleapis.com
    - Shared Gemini API client (reused across all calls)

    Result: first call connects to Gemini Live ~2–3× faster than cold start.
    """
    logger.info("Server starting — pre-warming Gemini services …")
    await prewarm_gemini()
    logger.info("All services ready. Accepting calls.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="Plivo Gemini Live Phone Bot", lifespan=lifespan)


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
    system_prompt: Optional[str] = None  # Per-call prompt override
    customer_name: Optional[str] = None  # Customer name injected into system prompt
    customer_id: Optional[int] = None    # Customer ID passed back in the webhook callback
    callback_url: Optional[str] = None   # URL to POST extracted lead data when call ends

@app.post("/call")
async def make_outbound_call(req: CallRequest):
    """Trigger an outbound call via Plivo REST API.

    Plivo dials `to`, and when answered it hits our /answer webhook which
    starts the Gemini Live audio stream — same flow as an inbound call.

    Example::
        curl -X POST http://localhost:8000/call \\
             -H 'Content-Type: application/json' \\
             -d '{"to": "+919876543210", "customer_id": 42, "customer_name": "Rahul", "callback_url": "https://your-service.com/lead"}'
    """
    auth_id = os.getenv("PLIVO_AUTH_ID")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN")
    from_number = req.from_ or os.getenv("PLIVO_FROM_NUMBER")
    ngrok_host = os.getenv("NGROK_HOST")

    if not auth_id or not auth_token:
        raise HTTPException(status_code=500, detail="PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN not set")
    if not from_number:
        raise HTTPException(status_code=500, detail="PLIVO_FROM_NUMBER not set in .env")
    if not ngrok_host:
        raise HTTPException(status_code=500, detail="NGROK_HOST not set in .env")

    from urllib.parse import quote
    params = []
    if req.customer_name: params.append(f"customer_name={quote(req.customer_name, safe='')}")
    if req.customer_id is not None: params.append(f"customer_id={req.customer_id}")
    if req.callback_url:  params.append(f"callback_url={quote(req.callback_url, safe='')}")
    # to_number is passed so the bot can include it in the callback payload
    params.append(f"to_number={quote(req.to, safe='')}")
    answer_url = f"https://{ngrok_host}/answer" + ("?" + "&".join(params) if params else "")
    # Plivo accepts E.164 with or without leading '+'; strip it to avoid format mismatches
    from_clean = from_number.lstrip("+")
    to_clean = req.to.lstrip("+")
    payload = {
        "from": from_clean,
        "to": to_clean,
        "answer_url": answer_url,
        "answer_method": "POST",
    }

    logger.info(f"Calling Plivo API | from={from_clean} to={to_clean} answer_url={answer_url}")
    endpoint = f"https://api.plivo.com/v1/Account/{auth_id}/Call/"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            endpoint,
            json=payload,
            auth=aiohttp.BasicAuth(auth_id, auth_token),
        ) as response:
            body = await response.json()
            if response.status not in (200, 201, 202):
                logger.error(f"Plivo outbound call failed: {response.status} {body}")
                raise HTTPException(status_code=response.status, detail=body)

    logger.info(f"Outbound call initiated | to={req.to} answer_url={answer_url}")
    return JSONResponse({"status": "calling", "to": req.to, "answer_url": answer_url})


@app.post("/answer")
async def answer_call(request: Request):
    """Plivo webhook — answers the call and streams audio to this server."""
    # NGROK_HOST takes priority; falls back to the request Host header
    host = os.getenv("NGROK_HOST") or request.headers.get("host", "yourdomain.com")
    ws_scheme = "wss" if os.getenv("USE_WSS", "true").lower() == "true" else "ws"

    from urllib.parse import quote
    qp = request.query_params
    params = []
    if qp.get("customer_name"): params.append(f"customer_name={quote(qp['customer_name'], safe='')}")
    if qp.get("customer_id"):   params.append(f"customer_id={qp['customer_id']}")
    if qp.get("callback_url"):  params.append(f"callback_url={quote(qp['callback_url'], safe='')}")
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
    customer_id: Optional[int] = None,
    callback_url: Optional[str] = None,
    to_number: Optional[str] = None,
):
    """WebSocket endpoint — one pipeline per connected call."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

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
            system_prompt=os.getenv("SYSTEM_PROMPT"),
            customer_name=customer_name,
            customer_id=customer_id,
            callback_url=callback_url,
            to_number=to_number,
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)