# Use Python 3.10 slim image for OCI runtime compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install only essential packages
RUN pip install --no-cache-dir fastapi uvicorn openai openenv-core pydantic aiohttp websockets

# Copy all files from root
COPY . .

# Crucial: Set PYTHONUNBUFFERED=1 to bypass all buffering
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "app.py"]
