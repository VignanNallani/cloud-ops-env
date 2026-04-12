FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn openai openenv-core pydantic aiohttp websockets
COPY . .
EXPOSE 8000
# -u flag ensures zero Python buffering
CMD ["python", "-u", "app.py"]
