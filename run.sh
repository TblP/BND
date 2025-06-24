#!/bin/bash

echo "==========================================="
echo "     Детекция людей с помощью Docker"
echo "==========================================="

# Создаем необходимые папки
echo "[1/5] Создание папок..."
mkdir -p input output models

# Копируем видео в input папку
echo "[2/5] Копирование видеофайла..."
if [ -f "input/crowd.mp4" ]; then
    cp crowd.mp4 input/
    echo "Видео crowd.mp4 скопировано в папку input"
else
    echo "ОШИБКА: Файл crowd.mp4 не найден!"
    exit 1
fi

# Проверяем Docker
echo "[3/5] Проверка Docker..."
if ! command -v docker &> /dev/null; then
    echo "ОШИБКА: Docker не найден! Установите Docker Desktop"
    exit 1
fi

# Собираем Docker образ
echo "[4/5] Сборка Docker образа..."
if ! docker build -t people-detector .; then
    echo "ОШИБКА: Не удалось собрать Docker образ"
    exit 1
fi

# Запускаем детекцию
echo "[5/5] Запуск детекции людей..."
if ! docker run --rm \
    -v "$(pwd)/input:/app/input:ro" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/models:/root/.cache/ultralytics" \
    people-detector \
    --input /app/input/crowd.mp4 \
    --output /app/output/result.mp4 \
    --confidence 0.3; then
    echo "ОШИБКА: Детекция завершилась с ошибкой"
    exit 1
fi

echo "==========================================="
echo "УСПЕХ! Проверьте файл output/result.mp4"
echo "==========================================="