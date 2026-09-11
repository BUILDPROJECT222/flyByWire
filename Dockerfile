# Hosted demo image (Railway): UI (vinext start) + worker (uvicorn) behind Caddy on $PORT.
# No GPU on Railway, so wgpu runs the connectome on Mesa's lavapipe (CPU Vulkan).

FROM node:22-bookworm-slim AS ui
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN VINEXT_NODE_SERVER=1 npx vinext build

FROM python:3.14-slim AS app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg mesa-vulkan-drivers libvulkan1 caddy ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ui /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY lab/ lab/
COPY data/ data/
# Download MaleCNS (~1.1 GB), prepare the graphs, then drop the raw feathers to keep the image small.
RUN python -m lab.download \
 && python -m lab.prepare \
 && python -m lab.prepare_full \
 && python -m lab.full_vision \
 && python -m lab.graded_engine \
 && rm data/malecns/*.feather
COPY experiments/active-quadratic-T4_T5.npz experiments/
RUN mkdir -p recordings && cp data/demo-default.mp4 recordings/demo-default.mp4
COPY --from=ui /app/ui ui/
COPY deploy/ deploy/
ENV PYTHONUNBUFFERED=1 FLYBYWIRE_IDLE_AFTER_S=30 PORT=8080
EXPOSE 8080
CMD ["bash", "deploy/start.sh"]
