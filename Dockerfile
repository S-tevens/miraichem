FROM node:22-slim AS web
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY configs ./configs
COPY results/published ./results/published
COPY --from=web /app/frontend/dist ./frontend/dist
ENV PORT=8000
EXPOSE 8000
# Hardware is off: the public copy must never be able to reach IBM. No IBM token is baked into the image.
CMD ["sh", "-c", "miraichem serve --host 0.0.0.0 --port ${PORT} --allow-remote --no-allow-hardware"]
