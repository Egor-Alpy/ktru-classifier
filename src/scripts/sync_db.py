import time
import schedule
from loguru import logger
import torch

from src.config import (
    EMBEDDING_MODEL, MODELS_DIR, DEVICE, BATCH_SIZE
)
from src.db import mongo_db, qdrant_db
from src.utils import (
    setup_logging, format_ktru_attributes,
    save_last_sync_time, get_last_sync_time
)
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct


def load_embedding_model():
    """Загрузка модели для создания эмбеддингов"""
    logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE,
        cache_folder=MODELS_DIR
    )
    logger.info(f"Модель эмбеддингов загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def prepare_ktru_text(item):
    """Подготовка текстового представления КТРУ кода"""
    title = item.get("title", "")
    description = item.get("description", "")
    attributes_text = format_ktru_attributes(item.get("attributes", []))

    combined_text = f"{title}. {description}. {attributes_text}"
    return combined_text


def sync_database():
    """Синхронизация базы данных КТРУ кодов с Qdrant"""
    logger.info("Запуск синхронизации базы данных")

    try:
        # Получение времени последней синхронизации
        last_sync = get_last_sync_time()

        if not last_sync:
            logger.warning("Не найдено время последней синхронизации, будет запущена полная индексация")
            from src.indexer import main as run_indexer
            run_indexer()
            return

        # Получение обновленных КТРУ кодов
        updated_items = mongo_db.get_latest_ktru_items(last_sync)

        if not updated_items:
            logger.info("Нет новых КТРУ кодов для обновления")
            save_last_sync_time()
            return

        logger.info(f"Найдено {len(updated_items)} обновленных КТРУ кодов")

        # Загрузка модели для создания эмбеддингов
        model = load_embedding_model()

        # Подготовка текстов для эмбеддингов
        texts = []
        metadata = []
        ids = []

        for i, item in enumerate(updated_items):
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

            # Генерация ID для записи в Qdrant
            ids.append(int(item.get("_id", {}).get("$oid", i)[-6:], 16))

        # Обработка по батчам
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_metadata = metadata[i:i + BATCH_SIZE]
            batch_ids = ids[i:i + BATCH_SIZE]

            # Создание эмбеддингов
            with torch.no_grad():
                embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                embeddings_list = embeddings.cpu().numpy().tolist()

            # Подготовка точек для загрузки в Qdrant
            points = [
                PointStruct(
                    id=id_,
                    vector=embedding,
                    payload=meta
                )
                for id_, embedding, meta in zip(batch_ids, embeddings_list, batch_metadata)
            ]

            # Загрузка точек в Qdrant
            qdrant_db.upload_points(points)

            logger.info(f"Обновлено {len(points)} КТРУ кодов")

        # Сохранение времени синхронизации
        save_last_sync_time()

        logger.info("Синхронизация базы данных завершена")
    except Exception as e:
        logger.error(f"Ошибка при синхронизации базы данных: {e}")


def main():
    """Основная функция для запуска планировщика синхронизации"""
    setup_logging()
    logger.info("Запуск планировщика синхронизации")

    # Запуск синхронизации при старте
    sync_database()

    # Планирование синхронизации каждые 24 часа
    schedule.every(24).hours.do(sync_database)

    # Бесконечный цикл для выполнения запланированных задач
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()