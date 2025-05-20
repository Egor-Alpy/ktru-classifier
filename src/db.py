from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict, Any, Union, Optional

from src.config import (
    MONGO_URI, MONGO_DB_NAME, KTRU_COLLECTION, PRODUCTS_COLLECTION,
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
)


class MongoDB:
    """Класс для работы с MongoDB"""

    def __init__(self):
        self.client: MongoClient = None
        self.db: Database = None
        self.ktru_collection: Collection = None
        self.products_collection: Collection = None
        self.connect()

    def connect(self):
        """Подключение к MongoDB"""
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[MONGO_DB_NAME]
            self.ktru_collection = self.db[KTRU_COLLECTION]
            self.products_collection = self.db[PRODUCTS_COLLECTION]
            logger.info(f"Успешное подключение к MongoDB: {MONGO_URI}")
        except Exception as e:
            logger.error(f"Ошибка подключения к MongoDB: {e}")
            raise

    def get_ktru_items(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение КТРУ кодов из базы данных"""
        try:
            return list(self.ktru_collection.find().skip(skip).limit(limit))
        except Exception as e:
            logger.error(f"Ошибка при получении КТРУ кодов: {e}")
            return []

    def count_ktru_items(self) -> int:
        """Подсчет количества КТРУ кодов в базе данных"""
        try:
            return self.ktru_collection.count_documents({})
        except Exception as e:
            logger.error(f"Ошибка при подсчете КТРУ кодов: {e}")
            return 0

    def get_ktru_item_by_code(self, ktru_code: str) -> Optional[Dict[str, Any]]:
        """Получение КТРУ кода по его значению"""
        try:
            return self.ktru_collection.find_one({"ktru_code": ktru_code})
        except Exception as e:
            logger.error(f"Ошибка при получении КТРУ кода {ktru_code}: {e}")
            return None

    def get_latest_ktru_items(self, timestamp: str) -> List[Dict[str, Any]]:
        """Получение КТРУ кодов, обновленных после указанной метки времени"""
        try:
            return list(self.ktru_collection.find({"updated_at": {"$gt": timestamp}}))
        except Exception as e:
            logger.error(f"Ошибка при получении последних КТРУ кодов: {e}")
            return []

    def close(self):
        """Закрытие соединения с MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Соединение с MongoDB закрыто")


class QdrantDB:
    """Класс для работы с Qdrant"""

    def __init__(self):
        self.client: QdrantClient = None
        self.connect()

    def connect(self):
        """Подключение к Qdrant"""
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            logger.info(f"Успешное подключение к Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            raise

    def create_collection(self, vector_size: int):
        """Создание коллекции в Qdrant"""
        try:
            # Проверка существования коллекции
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if COLLECTION_NAME in collection_names:
                logger.info(f"Коллекция {COLLECTION_NAME} уже существует")
                return

            # Создание коллекции
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Создана коллекция {COLLECTION_NAME} с размерностью векторов {vector_size}")
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {e}")
            raise

    def recreate_collection(self, vector_size: int):
        """Пересоздание коллекции в Qdrant"""
        try:
            # Удаление коллекции, если она существует
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if COLLECTION_NAME in collection_names:
                self.client.delete_collection(collection_name=COLLECTION_NAME)
                logger.info(f"Коллекция {COLLECTION_NAME} удалена")

            # Создание коллекции
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Пересоздана коллекция {COLLECTION_NAME} с размерностью векторов {vector_size}")
        except Exception as e:
            logger.error(f"Ошибка при пересоздании коллекции: {e}")
            raise

    def upload_points(self, points: List[PointStruct]):
        """Загрузка точек в Qdrant"""
        try:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            logger.info(f"Загружено {len(points)} точек в коллекцию {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке точек: {e}")
            raise

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Поиск похожих векторов в Qdrant"""
        try:
            search_results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit
            )
            return search_results
        except Exception as e:
            logger.error(f"Ошибка при поиске векторов: {e}")
            return []

    def close(self):
        """Закрытие соединения с Qdrant"""
        logger.info("Соединение с Qdrant закрыто")


# Создание синглтонов для доступа к базам данных
mongo_db = MongoDB()
qdrant_db = QdrantDB()
