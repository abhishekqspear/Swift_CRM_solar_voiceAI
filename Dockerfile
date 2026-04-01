# ── Pipecat Gemini Phone Bot ──────────────────────────────────────────────────
# Single image used for both bot_server (port 8000) and ui_server (port 8081).
# The CMD is overridden per service in docker-compose.yml.
#
# Build context must be the project root so the local pipecat/ checkout is
# included. Heavy layers (torch, pipecat) are ordered first to maximise cache hits.

FROM python:3.11-slim

# ── System libraries ──────────────────────────────────────────────────────────
# gcc/g++      : needed for packages with C extensions (websockets, aiohttp)
# libsndfile1  : libsndfile dependency for torchaudio
# libgomp1     : OpenMP runtime required by torch CPU kernels
# curl         : used by healthchecks in docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libsndfile1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Torch CPU-only ───────────────────────────────────────────────────
# Install before everything else so Docker cache is not busted by app changes.
# CPU-only wheels (~800 MB) save ~1.5 GB vs the default CUDA build.
# silero-vad only needs CPU inference on 8 kHz phone frames.
RUN pip install --no-cache-dir \
        "torch>=2.1.0" \
        "torchaudio>=2.1.0" \
        --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: Local pipecat checkout ──────────────────────────────────────────
# .dockerignore strips docs/, examples/, tests/, .git/ so only source is copied.
COPY pipecat/ ./pipecat/
RUN pip install --no-cache-dir "./pipecat[google,websocket,silero]"

# ── Layer 3: Remaining application dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 4: Application code ─────────────────────────────────────────────────
COPY bot.py server.py ui_server.py system_prompt.txt ./
COPY ui/ ./ui/

# Bake existing prompt files in as a seed directory.
# The entrypoint copies them into the mounted volume only if the volume is empty,
# so user edits made via the UI dashboard are never overwritten.
COPY prompts/ ./prompts_seed/
RUN mkdir -p prompts

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000 8081
ENTRYPOINT ["docker-entrypoint.sh"]
