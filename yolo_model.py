import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Tuple, List
import logging
from ultralytics import YOLO
import torch

class PeopleDetector:
    """
    Класс для детекции людей на видео с использованием YOLO модели.

    Attributes:
        model: Загруженная YOLO модель
        confidence_threshold: Порог уверенности для детекции
        person_class_id: ID класса "person" в COCO dataset
    """

    def __init__(self, model_path: str = 'yolo11m.pt', confidence_threshold: float = 0.5, device='auto'):
        """
        Инициализация детектора людей.

        Args:
            model_path: Путь к файлу модели YOLO
            confidence_threshold: Минимальный порог уверенности для детекции
        """
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0
        self.device = self._get_device(device)
        try:
            self.model = YOLO(model_path)
            if self.device != 'cpu':
                self.model.to(self.device)
            logging.info(f"Модель {model_path} загружена на {self.device}")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            raise

    def _get_device(self, device):
        """Определяет оптимальное устройство."""
        if device == 'auto':
            if torch.cuda.is_available():
                return f'cuda:{torch.cuda.current_device()}'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
            else:
                return 'cpu'
        return device
    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Выполняет детекцию людей на кадре.

        Args:
            frame: Входной кадр изображения

        Returns:
            Список кортежей (x1, y1, x2, y2, confidence) для каждого найденного человека
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Проверяем, что это человек и уверенность выше порога
                    if (int(box.cls[0]) == self.person_class_id and
                            float(box.conf[0]) >= self.confidence_threshold):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        detections.append((x1, y1, x2, y2, confidence))

        return detections

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Отрисовывает детекции на кадре.

        Args:
            frame: Исходный кадр
            detections: Список детекций людей

        Returns:
            Кадр с отрисованными детекциями
        """
        result_frame = frame.copy()

        for x1, y1, x2, y2, confidence in detections:
            # Рисуем прямоугольник вокруг человека
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Подготавливаем текст с классом и уверенностью
            label = f"person {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Рисуем фон для текста
            cv2.rectangle(result_frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          (0, 255, 0), -1)

            # Рисуем текст
            cv2.putText(result_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return result_frame

    def process_video(self, input_path: str, output_path: str) -> None:
        """
        Обрабатывает видеофайл, выполняя детекцию людей на каждом кадре.

        Args:
            input_path: Путь к входному видеофайлу
            output_path: Путь для сохранения обработанного видео
        """
        # Проверяем существование входного файла
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Видеофайл {input_path} не найден")

        # Открываем видеофайл
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл {input_path}")

        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Параметры видео: {width}x{height}, {fps} FPS, {total_frames} кадров")

        # Создаем объект для записи видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise ValueError(f"Не удалось создать выходной видеофайл {output_path}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Выполняем детекцию людей
                detections = self.detect_people(frame)

                # Отрисовываем детекции
                result_frame = self.draw_detections(frame, detections)

                # Записываем кадр
                out.write(result_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logging.info(f"Обработано {frame_count}/{total_frames} кадров ({progress:.1f}%)")

        finally:
            # Освобождаем ресурсы
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        logging.info(f"Обработка завершена. Результат сохранен в {output_path}")


def setup_logging() -> None:
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def parse_arguments() -> argparse.Namespace:
    """
    Парсинг аргументов командной строки.

    Returns:
        Объект с параметрами командной строки
    """
    parser = argparse.ArgumentParser(description='Детекция людей на видео')
    parser.add_argument('--input', '-i', required=True,
                        help='Путь к входному видеофайлу')
    parser.add_argument('--output', '-o', default='output/result.mp4',
                        help='Путь к выходному видеофайлу (по умолчанию: output/result.mp4)')
    parser.add_argument('--model', '-m', default='models/yolo11m.pt',
                        help='Путь к модели YOLO (по умолчанию: yolo11m.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                        help='Порог уверенности для детекции (по умолчанию: 0.5)')

    return parser.parse_args()


def main() -> None:
    """Основная функция программы."""
    setup_logging()

    try:
        args = parse_arguments()

        # Создаем детектор
        detector = PeopleDetector(
            model_path=args.model,
            confidence_threshold=args.confidence
        )

        # Обрабатываем видео
        detector.process_video(args.input, args.output)

        logging.info("Программа успешно завершена")

    except KeyboardInterrupt:
        logging.info("Программа прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Ошибка выполнения программы: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()