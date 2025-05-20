FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Установка Python и необходимых пакетов
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Порт для API
EXPOSE 8000

# Запуск API сервиса
CMD ["python3", "-m", "src.api"]
