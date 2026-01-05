FROM python:3.10-slim

# Instalăm ffmpeg pentru conversia audio
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

WORKDIR /app

# Copiem cerințele și instalăm
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiem codul sursă
COPY app ./app

# Portul expus
EXPOSE 8000

# Comanda de start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]