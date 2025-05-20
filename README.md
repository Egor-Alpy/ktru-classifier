# КТРУ-Классификатор

RAG-система для автоматической классификации товаров по кодам КТРУ (Каталог товаров, работ, услуг) с использованием Hugging Face моделей.

## Описание

Система использует векторную базу данных (Qdrant) для хранения эмбеддингов КТРУ кодов и их описаний. При поступлении нового товара, система создает его векторное представление, находит наиболее похожие КТРУ коды, и с помощью LLM-модели определяет оптимальный код КТРУ.

## Особенности

- Интеграция с MongoDB для получения данных о товарах и КТРУ кодах
- Векторный поиск для определения релевантных КТРУ кодов
- Использование современных языковых моделей для точной классификации
- API для интеграции с другими системами
- Автоматическая синхронизация с базой данных

## Требования

- Python 3.9+
- Docker и Docker Compose (опционально)
- GPU с минимум 16GB VRAM (рекомендуется)

## Установка и запуск

### Использование Docker

1. Клонируйте репозиторий:
git clone https://github.com/yourusername/ktru-classifier.git
cd ktru-classifier

2. Создайте файл `.env` на основе `.env.example`:
cp .env.example .env

3. Отредактируйте `.env` файл, указав корректные параметры подключения к MongoDB.

4. Запустите контейнеры:
docker-compose up -d

### Ручная установка

1. Клонируйте репозиторий:
git clone https://github.com/yourusername/ktru-classifier.git
cd ktru-classifier

2. Создайте и активируйте виртуальное окружение:
python -m venv venv
source venv/bin/activate  # Для Linux/Mac
venv\Scripts\activate     # Для Windows

3. Установите зависимости:
pip install -r requirements.txt

4. Создайте файл `.env` на основе `.env.example` и укажите параметры подключения.

5. Запустите индексацию КТРУ кодов:
python -m src.indexer

6. Запустите API сервер:
python -m src.api

## Использование API

### Классификация товара
POST /api/classify

Пример запроса:
```json
{
  "product": {
    "title": "Бумага туалетная \"Мягкий знак\" 1-слойная, С28, ст.32/72",
    "description": "Туалетная бумага «Мягкий знак» 1 сл, 1 рул, 100 % целлюлоза, 54 м.",
    "article": "100170",
    "brand": "Мягкий знак",
    "category": "Туалетная бумага",
    "attributes": [
      {
        "attr_name": "Тип",
        "attr_value": "Бытовая"
      },
      {
        "attr_name": "Количество слоев",
        "attr_value": "1"
      }
    ]
  }
}
Пример ответа:
json{
  "ktru_code": "17.12.20.111-00000001"
}
Лицензия
MIT

## requirements.txt
База
python-dotenv==1.0.0
pydantic==2.7.1
loguru==0.7.2
Работа с данными
pymongo==4.6.2
qdrant-client==1.8.0
faiss-gpu==1.7.2
numpy==1.26.4
pandas==2.2.1
Модели ML
torch==2.2.1
transformers==4.38.1
sentence-transformers==2.5.1
accelerate==0.23.0
API и сервер
fastapi==0.110.0
uvicorn==0.30.1
gunicorn==21.2.0
Утилиты
schedule==1.2.1
tqdm==4.66.2