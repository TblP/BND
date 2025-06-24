# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем только необходимые зависимости
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglib2.0-dev \
    libgl1-mesa-glx \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаем директорию для выходных файлов
RUN mkdir -p /app/output

# Создаем пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Точка входа по умолчанию
ENTRYPOINT ["python", "yolo_model.py"]

# Аргументы по умолчанию
CMD ["--help"]