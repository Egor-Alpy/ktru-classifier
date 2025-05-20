import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from typing import Dict, Any

from src.config import API_HOST, API_PORT
from src.models import ProductClassificationRequest, ProductClassificationResponse
from src.classifier import KTRUClassifier
from src.utils import setup_logging

# Настройка логирования
setup_logging()

# Инициализация API
app = FastAPI(
    title="КТРУ-Классификатор API",
    description="API для классификации товаров по кодам КТРУ с использованием RAG",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация классификатора
classifier = KTRUClassifier()


@app.get("/")
async def root():
    """Корневой маршрут API"""
    return {
        "message": "КТРУ-Классификатор API",
        "version": "1.0.0",
        "status": "OK"
    }


@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {
        "status": "OK",
        "classifier_loaded": classifier is not None
    }


@app.post("/api/classify", response_model=ProductClassificationResponse)
async def classify_product(request: ProductClassificationRequest):
    """Классификация товара по коду КТРУ"""
    try:
        logger.info(f"Получен запрос на классификацию товара: {request.product.title}")

        # Классификация товара
        ktru_code = classifier.classify(request.product.model_dump())

        logger.info(f"Товар '{request.product.title}' классифицирован с кодом КТРУ: {ktru_code}")

        return ProductClassificationResponse(ktru_code=ktru_code)
    except Exception as e:
        logger.error(f"Ошибка при классификации товара: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Основная функция для запуска API"""
    logger.info(f"Запуск API на {API_HOST}:{API_PORT}")
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()