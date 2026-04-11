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

# 7. Health check - SET TO 7860
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# EXPOSE both if necessary, but 7860 is your primary

EXPOSE 7860

# 8. Start server - SET TO 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
