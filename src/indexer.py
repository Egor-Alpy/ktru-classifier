import os
import json
import torch
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from typing import List, Dict, Any

from src.config import (
    BATCH_SIZE, EMBEDDING_MODEL, MODELS_DIR, DEVICE
)
from src.db import mongo_db, qdrant_db
from src.utils import setup_logging, format_ktru_attributes, save_last_sync_time


def load_embedding_model() -> SentenceTransformer:
    """Загрузка модели для создания эмбеддингов"""
    logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")

    # Создание директории для моделей, если она не существует
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Загрузка модели
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE,
        cache_folder=MODELS_DIR
    )
    logger.info(f"Модель эмбеддингов загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def prepare_ktru_text(item: Dict[str, Any]) -> str:
    """Подготовка текстового представления КТРУ кода"""
    title = item.get("title", "")
    description = item.get("description", "")
    attributes_text = format_ktru_attributes(item.get("attributes", []))

    combined_text = f"{title}. {description}. {attributes_text}"
    return combined_text


def create_embedding_for_ktru_items(model: SentenceTransformer) -> None:
    """Создание эмбеддингов для КТРУ кодов и сохранение их в Qdrant"""
    # Получение количества КТРУ кодов
    total_items = mongo_db.count_ktru_items()
    logger.info(f"Всего КТРУ кодов: {total_items}")

    # Создание/пересоздание коллекции в Qdrant
    vector_size = model.get_sentence_embedding_dimension()
    qdrant_db.recreate_collection(vector_size)

    # Обработка КТРУ кодов батчами
    for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Индексация КТРУ кодов"):
        # Получение батча КТРУ кодов
        batch = mongo_db.get_ktru_items(skip=i, limit=BATCH_SIZE)

        # Подготовка текстов для эмбеддингов
        texts = []
        metadata = []

        for item in batch:
            # Формирование текстового представления КТРУ
            combined_text = prepare_ktru_text(item)
            texts.append(combined_text)

            # Метаданные для сохранения
            metadata.append({
                "ktru_code": item.get("ktru_code", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "attributes": item.get("attributes", [])
            })

        # Создание эмбеддингов
        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            # Преобразование тензоров в список
            embeddings_list = embeddings.cpu().numpy().tolist()

        # Подготовка точек для загрузки в Qdrant
        points = [
            PointStruct(
                id=i + idx,
                vector=embedding,
                payload=meta
            )
            for idx, (embedding, meta) in enumerate(zip(embeddings_list, metadata))
        ]

        # Загрузка точек в Qdrant
        qdrant_db.upload_points(points)

    logger.info(f"Индексация завершена, всего проиндексировано {total_items} КТРУ кодов")
    save_last_sync_time()


def main():
    """Основная функция для запуска индексации"""
    setup_logging()
    logger.info("Запуск индексации КТРУ кодов")

    try:
        # Загрузка модели для создания эмбеддингов
        model = load_embedding_model()

        # Создание эмбеддингов для КТРУ кодов
        create_embedding_for_ktru_items(model)

        logger.info("Индексация успешно завершена")
    except Exception as e:
        logger.error(f"Ошибка при индексации: {e}")
        raise


if __name__ == "__main__":
    main()