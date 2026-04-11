ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# 3. Copy project
COPY . /app/

# 4. Install project
RUN uv pip install --system --no-cache -e .



# 5. Ensure runtime deps
RUN uv pip install --system fastapi uvicorn

# 6. Environment variables
ENV PYTHONPATH="/app"
ENV MODULE_PATH="server.app:app"

# 7. Health check - SET TO 8000
# Using 0.0.0.0 is often more reliable than localhost in some Docker networks
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://0.0.0.0:8000/health || exit 1

# 8. Start server - SET TO 8000
# This ensures Point 7 and Point 8 are on the SAME DOOR.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
