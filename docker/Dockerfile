FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & python app/gradio_app.py"]