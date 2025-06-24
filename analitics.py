import cv2
import numpy as np
from ultralytics import YOLO


class SimpleAnalyzer:
    """Простой анализатор качества детекции людей."""

    def __init__(self, model_path: str = 'models/yolo11m.pt', confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0

    def analyze_video(self, video_path: str, max_frames: int = 100) -> dict:
        """
        Анализирует видео и возвращает основные метрики.

        Args:
            video_path: Путь к видеофайлу
            max_frames: Максимальное количество кадров для анализа

        Returns:
            Словарь с результатами анализа
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)

        people_counts = []
        confidences = []
        frames_analyzed = 0
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret or frames_analyzed >= max_frames:
                break

            if frame_number % step == 0:
                results = self.model(frame, verbose=False)

                frame_people = 0
                frame_confidences = []

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if (int(box.cls[0]) == self.person_class_id and
                                    float(box.conf[0]) >= self.confidence_threshold):
                                frame_people += 1
                                frame_confidences.append(float(box.conf[0]))

                people_counts.append(frame_people)
                if frame_confidences:
                    confidences.extend(frame_confidences)

                frames_analyzed += 1

            frame_number += 1

        cap.release()

        # Вычисляем метрики
        avg_people = np.mean(people_counts) if people_counts else 0
        max_people = max(people_counts) if people_counts else 0
        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            'Среднее кол-во людей на кадре': round(avg_people, 1),
            'Максимальное кол-во людей на кадре': max_people,
            'Средняя точность': round(avg_confidence, 3)
        }


def main():
    """Пример использования анализатора."""
    analyzer = SimpleAnalyzer(model_path='models/yolo11m.pt', confidence_threshold=0.5)
    video_path = "input/crowd.mp4"

    try:
        results = analyzer.analyze_video(video_path, max_frames=50)
        for key, value in results.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()