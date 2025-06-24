@echo off
chcp 65001 >nul

echo ===========================================
echo     Детекция людей с помощью Docker
echo ===========================================

REM Создаем необходимые папки
echo [1/5] Создание папок...
mkdir input 2>nul
mkdir output 2>nul
mkdir models 2>nul

REM Копируем видео в input папку
echo [2/5] Копирование видеофайла...
if exist "input/crowd.mp4" (
    copy crowd.mp4 input\ >nul
    echo Видео crowd.mp4 скопировано в папку input
) else (
    echo ОШИБКА: Файл crowd.mp4 не найден!
    pause
    exit /b 1
)

REM Проверяем Docker
echo [3/5] Проверка Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Docker не найден! Установите Docker Desktop
    pause
    exit /b 1
)

REM Собираем Docker образ
echo [4/5] Сборка Docker образа...
docker build -t people-detector .
if errorlevel 1 (
    echo ОШИБКА: Не удалось собрать Docker образ
    pause
    exit /b 1
)

REM Запускаем детекцию
echo [5/5] Запуск детекции людей...
docker run --rm ^
    -v "%cd%\input:/app/input:ro" ^
    -v "%cd%\output:/app/output" ^
    -v "%cd%\models:/root/.cache/ultralytics" ^
    people-detector ^
    --input /app/input/crowd.mp4 ^
    --output /app/output/result.mp4 ^
    --confidence 0.5

if errorlevel 1 (
    echo ОШИБКА: Детекция завершилась с ошибкой
    pause
    exit /b 1
)

echo ===========================================
echo УСПЕХ! Проверьте файл output\result.mp4
echo ===========================================
pause