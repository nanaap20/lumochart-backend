# Full Python image ensures all build tools and audio libs available
FROM python:3.11

# Install ffmpeg for Whisper and audio ops
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Prevent pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# App directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port
EXPOSE 8080

# Gunicorn gracefully handles multiple uvicorn workers
CMD ["sh", "-c", "gunicorn -w 2 -k uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:$PORT main:app"]
