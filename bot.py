#
# Plivo + Gemini Live Phone Call Bot
# Voice-to-voice with per-call context isolation and speech interruption
#

import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import ActivityEnd, ActivityStart, Blob, HttpOptions, ThinkingConfig
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

load_dotenv(override=True)

# How much audio to pre-buffer before Silero fires (captures utterance start).
# Silero needs start_secs=0.2s of speech before triggering, so buffering
# ~300ms gives us comfortable coverage for the beginning of each utterance.
_PRE_BUFFER_MS = 300


# ── Shared Google AI client ────────────────────────────────────────────────────

_shared_gemini_client: Optional[Client] = None


def get_shared_client() -> Client:
    global _shared_gemini_client
    if _shared_gemini_client is None:
        _shared_gemini_client = Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=HttpOptions(api_version="v1beta"),
        )
        logger.info("Gemini API client created (shared across calls)")
    return _shared_gemini_client


async def prewarm_gemini():
    """Pre-warm DNS + TLS and initialise the shared Gemini client at startup."""
    import aiohttp

    get_shared_client()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://generativelanguage.googleapis.com/",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                logger.info(
                    f"Google API endpoint reachable (HTTP {resp.status}) — DNS + TLS pre-warmed"
                )
    except Exception as e:
        logger.warning(f"Gemini pre-warm ping failed (non-fatal): {e}")

    logger.info("Gemini pre-warm complete — server ready to handle calls")


# ── Service subclass: shared client + telephony turn management ───────────────

class _PhoneBotGeminiService(GeminiLiveLLMService):
    """GeminiLiveLLMService customised for telephony.

    Key overrides
    -------------
    1. ``create_client`` — injects the process-wide shared Client.

    2. ``_handle_session_ready`` — sends the initial greeting as soon as the
       Gemini session is established, avoiding the race condition where
       on_client_connected fires before the session is ready.

    3. ``_send_user_audio`` — implements pre-buffered, gated audio delivery:
       - While the user is NOT speaking: accumulate a rolling 300 ms ring
         buffer but DO NOT forward audio to Gemini (prevents noise
         accumulation that causes growing TTFB per turn).
       - When the user starts speaking: flush the pre-buffer first (so Gemini
         hears the beginning of the utterance that Silero's 200 ms onset
         window would otherwise miss), then forward all subsequent frames.
       - When the user stops speaking: stop forwarding until next turn.

    4. ``_handle_user_started_speaking`` / ``_handle_user_stopped_speaking`` —
       send ``activity_start`` / ``activity_end`` via ``send_realtime_input``
       (the correct API when server-side VAD is disabled; ``send_client_content``
       cannot be mixed with the realtime-input audio stream).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_sending_audio = False
        self._pre_buffer: list = []          # [(audio_bytes, sample_rate), …]
        self._pre_buffer_bytes = 0
        self._pre_buffer_max_bytes = 0       # computed on first audio frame
        self._transcript: list[dict] = []    # [{"role": "user"|"bot", "text": "..."}, …]
        self._bot_turn_buffer: str = ""      # accumulates output transcription within a turn
        self._session_ready_time: float = 0.0  # monotonic time when Gemini session became ready
        # Pre-rendered greeting (played instantly on connect to mask Gemini cold-start)
        self._greeting_prerendered: bool = False  # True when a TTS greeting clip will be / was played
        self._greeting_text: Optional[str] = None  # text of the pre-rendered greeting (seeded into history)
        self._greeting_started_time: float = 0.0  # monotonic time the greeting clip was queued
        self._first_chunk_logged: bool = False  # one-shot guard for first-audio latency log
        self._connect_start_time: float = 0.0  # monotonic time the Gemini connect began

    def create_client(self):
        self._client = get_shared_client()

    async def _connect(self, session_resumption_handle=None):
        """Record connect-start for session-open latency instrumentation."""
        if not self._connect_start_time:
            self._connect_start_time = time.monotonic()
        await super()._connect(session_resumption_handle=session_resumption_handle)

    async def _push_user_transcription(self, text, result=None):
        """Capture complete user sentences into the transcript."""
        await super()._push_user_transcription(text, result=result)
        if text:
            self._transcript.append({"role": "user", "text": text})

    async def _handle_msg_output_transcription(self, message):
        """Accumulate bot output text within the current turn."""
        await super()._handle_msg_output_transcription(message)
        text = message.server_content.output_transcription.text if message.server_content.output_transcription else ""
        if text:
            self._bot_turn_buffer += text

    async def _handle_msg_turn_complete(self, message):
        """Flush accumulated bot turn text to transcript on turn end.

        Also flush any user transcription text stuck in the parent's buffer
        (Gemini doesn't always send an end-of-sentence marker over phone audio,
        leaving partial utterances that never reach _push_user_transcription).
        """
        # Flush parent's user transcription buffer before calling super()
        # so the user turn is captured before the bot turn is flushed.
        leftover = getattr(self, "_user_transcription_buffer", "").strip()
        if leftover:
            self._user_transcription_buffer = ""
            self._transcript.append({"role": "user", "text": leftover})

        await super()._handle_msg_turn_complete(message)

        if self._bot_turn_buffer:
            self._transcript.append({"role": "bot", "text": self._bot_turn_buffer.strip()})
            self._bot_turn_buffer = ""

    async def _handle_session_ready(self, session):
        """Trigger greeting once Gemini session is live.

        send_client_content is used here intentionally — it must fire
        immediately (no sleep) so it always arrives BEFORE any Silero
        activity_start.  The 1-second sleep that was tried previously caused
        random non-responses because it let activity_start arrive mid-greeting,
        injecting send_client_content into an already-open audio turn.
        """
        await super()._handle_session_ready(session)
        self._session_ready_time = time.monotonic()
        if self._connect_start_time:
            logger.info(
                f"[latency] connect_start→session_ready "
                f"{(self._session_ready_time - self._connect_start_time) * 1000:.0f}ms "
                f"(Gemini session open + system_instruction parse)"
            )
        if self._session and not self._disconnecting:
            try:
                if self._greeting_prerendered:
                    # Greeting was already spoken via the pre-rendered clip, and the
                    # live model's system instruction already tells it the greeting
                    # happened (see run_bot). So send NOTHING here — no live greeting
                    # trigger — and let the caller's first utterance drive turn 2.
                    logger.info("Pre-rendered greeting played; live greeting trigger suppressed")
                else:
                    from google.genai.types import Content, Part
                    greeting_turn = Content(
                        role="user",
                        parts=[Part(text="The call has connected. Begin the conversation per your instructions.")],
                    )
                    await self._session.send_client_content(turns=[greeting_turn], turn_complete=True)
                    logger.debug("Sent greeting trigger to Gemini")
                logger.info(
                    f"[latency] session_ready→greeting_sent "
                    f"{(time.monotonic() - self._session_ready_time) * 1000:.0f}ms"
                )
            except Exception as e:
                logger.warning(f"Greeting trigger failed: {e}")

    async def _handle_msg_model_turn(self, msg):
        """Instrumentation: log latency from session-ready to the first model turn.

        In the live-greeting path this is the greeting generation TTFB; in the
        pre-rendered path it is the response to the caller's first utterance.
        """
        if not self._first_chunk_logged and self._session_ready_time:
            self._first_chunk_logged = True
            logger.info(
                f"[latency] session_ready→first_model_turn "
                f"{(time.monotonic() - self._session_ready_time) * 1000:.0f}ms"
            )
        await super()._handle_msg_model_turn(msg)

    async def _send_user_audio(self, frame):
        """Pre-buffer audio; only send to Gemini during active speech windows."""
        if self._disconnecting or not self._session:
            return

        # Compute ring-buffer size on the first frame we see.
        if not self._pre_buffer_max_bytes:
            bytes_per_ms = (frame.sample_rate * frame.num_channels * 2) // 1000
            self._pre_buffer_max_bytes = bytes_per_ms * _PRE_BUFFER_MS

        if not self._is_sending_audio:
            # Maintain a rolling pre-buffer; discard oldest frames when full.
            self._pre_buffer.append((bytes(frame.audio), frame.sample_rate))
            self._pre_buffer_bytes += len(frame.audio)
            while self._pre_buffer_bytes > self._pre_buffer_max_bytes and self._pre_buffer:
                removed, _ = self._pre_buffer.pop(0)
                self._pre_buffer_bytes -= len(removed)
            return

        # Forwarding mode — send frame to Gemini.
        try:
            await self._session.send_realtime_input(
                audio=Blob(data=frame.audio, mime_type=f"audio/pcm;rate={frame.sample_rate}")
            )
        except Exception as e:
            await self._handle_send_error(e)

    async def _handle_user_started_speaking(self, frame):
        # Ignore interruptions during the startup grace period so the bot can
        # deliver its greeting before the user's "hello" on pickup cuts it off.
        # The window is active if EITHER the session just became ready OR a
        # pre-rendered greeting clip just started playing — the latter happens
        # BEFORE the session is ready, so keying off session_ready alone would
        # let a pre-roll "hello" bypass the grace period.
        grace_secs = float(os.getenv("STARTUP_GRACE_SECS", "4.0"))
        now = time.monotonic()
        in_grace = (
            (self._session_ready_time and (now - self._session_ready_time) < grace_secs)
            or (self._greeting_started_time and (now - self._greeting_started_time) < grace_secs)
        )
        if in_grace:
            logger.debug(f"Ignoring speech start — within {grace_secs}s startup grace period")
            return

        self._user_is_speaking = True
        # Snapshot and clear pre-buffer BEFORE any awaits so that audio frames
        # arriving while we yield to the event loop go directly to Gemini
        # (not into a pre-buffer that we're about to discard).
        pre_buf = list(self._pre_buffer)
        self._pre_buffer = []
        self._pre_buffer_bytes = 0
        self._is_sending_audio = True  # gate open — live frames go straight to Gemini
        if self._session and not self._disconnecting:
            try:
                await self._session.send_realtime_input(activity_start=ActivityStart())
                for audio_bytes, sample_rate in pre_buf:
                    await self._session.send_realtime_input(
                        audio=Blob(data=audio_bytes, mime_type=f"audio/pcm;rate={sample_rate}")
                    )
                logger.debug(f"activity_start + flushed {len(pre_buf)} pre-buffer frames")
            except Exception as e:
                logger.warning(f"activity_start/pre-buffer flush failed: {e}")

    async def _handle_user_stopped_speaking(self, frame):
        self._user_is_speaking = False
        self._user_audio_buffer = bytearray()
        # Close the gate BEFORE the await so no stray audio frames slip through.
        self._is_sending_audio = False
        self._pre_buffer = []
        self._pre_buffer_bytes = 0
        await self.start_ttfb_metrics()
        if self._session and not self._disconnecting:
            try:
                await self._session.send_realtime_input(activity_end=ActivityEnd())
                logger.debug("Sent activity_end to Gemini (user stopped speaking via Silero)")
            except Exception as e:
                logger.warning(f"activity_end send failed: {e}")


# ── Turn signal converter ──────────────────────────────────────────────────────

class _TurnSignalConverter(FrameProcessor):
    """Converts Silero VAD frames to the speaking frames GeminiLive understands.

    ``VADProcessor`` emits ``VADUserStartedSpeakingFrame`` /
    ``VADUserStoppedSpeakingFrame``.  GeminiLive's ``process_frame`` handles
    ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` (no "VAD"
    prefix) — those are the ones that update its internal speaking state and
    trigger the activity handlers.  This converter bridges the gap.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.push_frame(UserStartedSpeakingFrame(), direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.push_frame(UserStoppedSpeakingFrame(), direction)

        await self.push_frame(frame, direction)


# ── Early Interruption Processor ──────────────────────────────────────────────

class EarlyInterruptor(FrameProcessor):
    """Clears Plivo's audio buffer the moment caller speech is detected —
    but ONLY while the bot is actively speaking.

    Tuning via .env
    ---------------
    INTERRUPTION_ENERGY_THRESHOLD  RMS threshold (default 600).
        Phone speech is typically 500–3000 RMS; line noise is below 300 RMS.
    """

    def __init__(
        self,
        stream_id: str,
        energy_threshold: int = 600,
        hold_frames: int = 3,
        cooldown_frames: int = 50,
        echo_guard_frames: int = 30,   # ~600ms at 20ms/frame — ignore echo at turn start
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stream_id = stream_id
        self._threshold = energy_threshold
        self._hold = hold_frames
        self._cooldown = cooldown_frames
        self._echo_guard = echo_guard_frames
        self._bot_speaking = False
        self._high_energy_count = 0
        self._cooldown_count = 0
        self._frames_since_armed = 0
        self._fired = False
        # Diagnostic: count and sample energy of frames received after bot stops
        self._diag_frame_count = 0
        self._diag_max_rms = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
                self._fired = False
                self._high_energy_count = 0
                self._cooldown_count = 0
                self._frames_since_armed = 0
                logger.debug("EarlyInterruptor: armed (bot started speaking)")
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                self._fired = False
                self._diag_frame_count = 0
                self._diag_max_rms = 0.0
                logger.debug("EarlyInterruptor: disarmed (bot stopped speaking)")

        elif isinstance(frame, InputAudioRawFrame) and not self._bot_speaking:
            # Diagnostic: log audio energy every 50 frames (~1s) after bot stops
            self._diag_frame_count += 1
            audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
            if rms > self._diag_max_rms:
                self._diag_max_rms = rms
            if self._diag_frame_count % 50 == 0:
                logger.debug(
                    f"EarlyInterruptor [idle] frame#{self._diag_frame_count} "
                    f"peak_rms={self._diag_max_rms:.0f} (threshold={self._threshold})"
                )
                self._diag_max_rms = 0.0

        elif isinstance(frame, InputAudioRawFrame) and self._bot_speaking:
            self._frames_since_armed += 1
            if self._frames_since_armed <= self._echo_guard:
                # Within the echo guard window — the bot just started speaking
                # and any high-energy audio is likely echo of its own voice
                # bouncing back through the phone line.  Skip detection.
                await self.push_frame(frame, direction)
                return

            audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

            if rms > self._threshold:
                self._high_energy_count += 1
                self._cooldown_count = 0
                if self._high_energy_count >= self._hold and not self._fired:
                    self._fired = True
                    logger.debug(
                        f"EarlyInterruptor: caller speech (rms={rms:.0f}) "
                        f"during bot speech → clearAudio"
                    )
                    await self.push_frame(
                        OutputTransportMessageUrgentFrame(
                            message={"event": "clearAudio", "streamId": self._stream_id}
                        ),
                        FrameDirection.DOWNSTREAM,
                    )
            else:
                self._high_energy_count = 0
                self._cooldown_count += 1
                if self._cooldown_count >= self._cooldown:
                    self._fired = False

        await self.push_frame(frame, direction)


# ── Lead extraction helpers ────────────────────────────────────────────────────

async def extract_lead_fields(transcript: list[dict]) -> dict:
    """Send transcript to Gemini REST and extract structured lead fields as JSON."""
    text = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in transcript)

    # Give Gemini a concrete "now" so it can resolve relative phrases like
    # "in ten minutes" / "kal" / "at 3pm" into an absolute ISO 8601 datetime
    # in the caller's local timezone. CALLBACK_TZ defaults to IST for the
    # current Indian dev/QA — set CALLBACK_TZ=Australia/Sydney for AU prod.
    tz_name = os.environ.get("CALLBACK_TZ", "Asia/Kolkata")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")
    now_local = datetime.now(tz)
    now_iso = now_local.isoformat(timespec="seconds")

    prompt = (
        f"Current time is {now_iso} (timezone: {tz_name}). "
        "Extract the following fields from this sales call transcript. "
        "Return ONLY valid JSON (no markdown, no explanation) with exactly these keys:\n"
        "- full_name: string\n"
        "- location: string or 'Unknown'\n"
        "- property_type: 'Residential' | 'Commercial' | 'Industrial' | 'Unknown'\n"
        "- monthly_bill: exact words used by caller (e.g. '5000 rupees') or 'Unknown'\n"
        "- roof_type: 'Concrete' | 'Metal' | 'Other' | 'Unknown'\n"
        "- ownership: 'Own' | 'Rented' | 'Unknown'\n"
        "- intent: 'INTERESTED' | 'EXPLORING' | 'NOT_INTERESTED' | 'CALLBACK' | 'DO_NOT_CALL' | 'UNKNOWN'\n"
        "- interest_level: 'HOT' | 'WARM' | 'COLD' | 'UNKNOWN'\n"
        "- qualification: 'QUALIFIED' | 'UNQUALIFIED' | 'UNKNOWN'\n"
        "- callback_time: ABSOLUTE ISO 8601 datetime with timezone offset (e.g. "
        f"'{now_iso}') if caller mentioned any callback timing, otherwise null. "
        "Convert relative phrases using the current time above:\n"
        f"  * 'in 10 minutes' / '10 minute me' / 'after ten minutes' → current time + 10 min\n"
        f"  * 'in 2 hours' / '2 ghante me' → current time + 2 hr\n"
        f"  * 'tomorrow' / 'kal' (no specific time) → next day, same hour\n"
        f"  * 'at 3 pm' / '3 baje' / '15:00' → today at 15:00 in {tz_name}, "
        "or tomorrow if 15:00 has already passed today\n"
        f"  * 'tomorrow at 9 am' / 'kal 9 baje' → next day at 09:00 in {tz_name}\n"
        "  Always return a fully-qualified ISO datetime with the timezone offset; "
        "never a bare time string. Return null only if no timing was mentioned.\n"
        "\nTranscript:\n" + text
    )
    try:
        client = get_shared_client()
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = response.text.strip()
        # Strip markdown code fence if Gemini wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.error(f"Lead extraction failed: {e}")
        return {k: "UNKNOWN" for k in [
            "full_name", "location", "property_type", "monthly_bill",
            "roof_type", "ownership", "intent", "interest_level", "qualification",
        ]} | {"callback_time": None}


async def send_callback(
    callback_url: str,
    lead_id: Optional[str],
    call_id: Optional[str],
    phone_number: Optional[str],
    fields: dict,
    transcript: list[dict],
    call_duration_seconds: int = 0,
):
    """POST extracted lead data to the calling service's callback URL."""
    transcript_text = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in transcript)
    payload = {
        "lead_id": lead_id,
        "call_id": call_id,
        "phone_number": phone_number,
        "call_duration_seconds": call_duration_seconds,
        "fields": {
            "full_name":      fields.get("full_name", "Unknown"),
            "location":       fields.get("location", "Unknown"),
            "property_type":  fields.get("property_type", "Unknown"),
            "monthly_bill":   fields.get("monthly_bill", "Unknown"),
            "roof_type":      fields.get("roof_type", "Unknown"),
            "ownership":      fields.get("ownership", "Unknown"),
            "intent":         fields.get("intent", "UNKNOWN"),
            "interest_level": fields.get("interest_level", "UNKNOWN"),
            "qualification":  fields.get("qualification", "UNKNOWN"),
            "callback_time":  fields.get("callback_time", None),
        },
        "transcript": transcript_text,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                logger.info(f"Lead callback → {callback_url} HTTP {resp.status} | lead_id={lead_id}")
    except Exception as e:
        logger.error(f"Lead callback failed (url={callback_url}): {e}")


# ── Prompt trimming for the live model ──────────────────────────────────────────

# Sections that only matter for POST-call lead extraction (done separately by
# extract_lead_fields on the transcript). They add nothing to the live
# conversation but are billed as input tokens on every turn — so strip them from
# the prompt sent to the live model. The bot still gets role, style, info to
# collect, intent + DO_NOT_CALL triggers, conversation flow, and rules.
_LIVE_PROMPT_DROP_SECTIONS = ("OUTPUT FIELDS", "INTEREST LEVEL CLASSIFICATION")
_SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 /()\-]*:$")


def trim_prompt_for_live(prompt: str) -> str:
    """Drop post-call-only sections from the system prompt sent to the live model.

    Section headers are UPPERCASE lines ending in ':'. A dropped section runs
    until the next recognized header. Unrecognized prompts are returned
    unchanged, so this is safe for arbitrary prompt text.
    """
    if not prompt:
        return prompt
    out: list[str] = []
    skipping = False
    for line in prompt.split("\n"):
        stripped = line.strip()
        if _SECTION_HEADER_RE.match(stripped):
            skipping = stripped[:-1].strip() in _LIVE_PROMPT_DROP_SECTIONS
        if not skipping:
            out.append(line)
    return "\n".join(out).strip()


# ── Bot entry point ────────────────────────────────────────────────────────────

GEMINI_VOICES = {"Puck", "Aoede", "Charon", "Fenrir", "Kore", "Leda", "Orus", "Zephyr"}

async def run_bot(
    websocket,
    stream_id: str,
    call_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    customer_name: Optional[str] = None,
    lead_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    to_number: Optional[str] = None,
    voice: Optional[str] = None,
    greeting_pcm: Optional[bytes] = None,
    greeting_rate: Optional[int] = None,
    greeting_text: Optional[str] = None,
):
    """Run the Gemini Live bot for a single Plivo call."""

    _call_start = time.monotonic()

    # Load system prompt: explicit arg > .env SYSTEM_PROMPT > system_prompt.txt > hardcoded default.
    # Only touch disk when neither the per-call arg nor the env override is set —
    # the synchronous read otherwise runs on the event loop on the hot connect path.
    _file_prompt = ""
    if not system_prompt and not os.getenv("SYSTEM_PROMPT"):
        _prompt_file = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
        try:
            with open(_prompt_file, "r", encoding="utf-8") as _f:
                _file_prompt = _f.read().strip()
        except Exception as _e:
            logger.warning(f"Could not read system_prompt.txt: {_e}")

    effective_prompt = (
        system_prompt
        or os.getenv("SYSTEM_PROMPT")
        or _file_prompt
        or 'You are a helpful assistant. Greet the caller and assist them.'
    )
    effective_prompt = effective_prompt.replace("{customer_name}", customer_name or "the caller")

    # Strip post-call-only sections (lead-extraction taxonomy / output schema) so
    # the live model isn't billed for them every turn. extract_lead_fields handles
    # that classification separately from the transcript after the call.
    effective_prompt = trim_prompt_for_live(effective_prompt)

    # When a pre-rendered greeting is played on connect, tell the live model the
    # greeting ALREADY happened — otherwise it follows its START instruction and
    # greets a second time after the caller's first reply.
    if greeting_pcm and greeting_text:
        effective_prompt += (
            "\n\nCONVERSATION ALREADY STARTED:\n"
            f'You have ALREADY greeted the caller out loud with this exact line: "{greeting_text}"\n'
            "Do NOT greet again. Do NOT repeat your name, your company, or your opening question. "
            "The caller is now responding to that greeting — reply directly and naturally to what they say next."
        )

    logger.info(f"System prompt source: {'arg' if system_prompt else 'env' if os.getenv('SYSTEM_PROMPT') else 'file' if _file_prompt else 'default'} ({len(effective_prompt)} chars) customer_name={customer_name!r} prerendered_greeting={bool(greeting_pcm)}")

    logger.info(f"Starting bot | stream_id={stream_id} call_id={call_id}")

    # ── Serializer ────────────────────────────────────────────────────────────
    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=os.getenv("PLIVO_AUTH_ID"),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN"),
        params=PlivoFrameSerializer.InputParams(auto_hang_up=True),
    )

    # ── Transport ─────────────────────────────────────────────────────────────
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
        ),
    )

    # ── Silero VAD — detects speech/silence on phone audio (reliable at 8→16kHz)
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(
            confidence=float(os.getenv("VAD_CONFIDENCE", "0.5")),  # raise to 0.5–0.7 to reduce noise triggers
            stop_secs=float(os.getenv("VAD_STOP_SECS", "0.5")),   # 500ms silence → end of turn
            start_secs=float(os.getenv("VAD_START_SECS", "0.3")), # 300ms sustained speech required to trigger
            min_volume=0.0,   # disable EBU R128 volume gate — phone audio at 8kHz
                              # measures very low on that scale; rely on Silero only
        ))
    )

    # ── VAD frame converter ───────────────────────────────────────────────────
    turn_converter = _TurnSignalConverter()

    # ── Early interruptor ─────────────────────────────────────────────────────
    early_interruptor = EarlyInterruptor(
        stream_id=stream_id,
        energy_threshold=int(os.getenv("INTERRUPTION_ENERGY_THRESHOLD", "600")),
    )

    # ── Gemini Live ───────────────────────────────────────────────────────────
    logger.debug(f"system_instruction preview: {effective_prompt[:120]!r}")
    # Optional model override. Unset → pipecat's default native-audio model.
    # Set GEMINI_LIVE_MODEL to try a lower-latency half-cascade model, e.g.
    # "gemini-2.0-flash-live-001" or "gemini-live-2.5-flash-preview".
    _settings_kwargs = dict(
        voice=voice or os.getenv("GEMINI_VOICE", "Puck"),  # Aoede | Charon | Fenrir | Kore | Leda | Orus | Puck | Zephyr
        vad=GeminiVADParams(disabled=True),  # Silero owns all VAD
        thinking=ThinkingConfig(thinking_budget=0),  # disable reasoning → low latency
    )
    _live_model = os.getenv("GEMINI_LIVE_MODEL")
    if _live_model:
        _settings_kwargs["model"] = _live_model
        logger.info(f"Using GEMINI_LIVE_MODEL override: {_live_model}")
    llm = _PhoneBotGeminiService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=effective_prompt,  # must be direct param — Settings path skips _system_instruction_from_init
        settings=GeminiLiveLLMService.Settings(**_settings_kwargs),
    )

    # Wire pre-rendered greeting: when a clip is available it is played on connect
    # and its text is seeded into Gemini history (live greeting suppressed).
    if greeting_pcm:
        llm._greeting_prerendered = True
        llm._greeting_text = greeting_text

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            turn_converter,
            early_interruptor,
            llm,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Events ────────────────────────────────────────────────────────────────

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected | stream_id={stream_id}")
        # Play the pre-rendered greeting the instant the call connects so the
        # caller hears the bot immediately, while the Gemini Live session warms
        # up in the background. Gemini takes over from the caller's first reply.
        if greeting_pcm:
            try:
                rate = greeting_rate or 24000
                await task.queue_frames([
                    TTSStartedFrame(),
                    TTSAudioRawFrame(audio=greeting_pcm, sample_rate=rate, num_channels=1),
                    TTSStoppedFrame(),
                ])
                # Mark grace window start (audio plays BEFORE session is ready) and
                # record the greeting in the transcript — a TTS clip produces no
                # Gemini output_transcription, so lead extraction would miss it.
                llm._greeting_started_time = time.monotonic()
                if greeting_text:
                    llm._transcript.append({"role": "bot", "text": greeting_text})
                logger.info(
                    f"[latency] played pre-rendered greeting "
                    f"({len(greeting_pcm)} bytes @ {rate}Hz) on connect | stream_id={stream_id}"
                )
            except Exception as e:
                logger.warning(f"Pre-rendered greeting playback failed (falling back to live): {e}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected | stream_id={stream_id}")
        await task.cancel()
        if callback_url:
            logger.info(f"Extracting lead fields from {len(llm._transcript)} transcript turns")
            fields = await extract_lead_fields(llm._transcript)
            # customer_name is pre-known — use it if extraction didn't find a name
            if customer_name and fields.get("full_name") in (None, "Unknown", "UNKNOWN", ""):
                fields["full_name"] = customer_name
            await send_callback(
                callback_url, lead_id, call_id, to_number, fields, llm._transcript,
                call_duration_seconds=int(time.monotonic() - _call_start),
            )

    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.warning(f"Session timeout | stream_id={stream_id}")
        await task.cancel()

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info(f"Bot finished | stream_id={stream_id}")