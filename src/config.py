import os
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ktru_database")
KTRU_COLLECTION = os.getenv("KTRU_COLLECTION", "ktru_collection")
PRODUCTS_COLLECTION = os.getenv("PRODUCTS_COLLECTION", "products_collection")

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ktru_embeddings")

# Модели Hugging Face
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ai-forever/sbert_large_nlu_ru")
LLM_MODEL = os.getenv("LLM_MODEL", "ai-forever/rugpt-3.5-13b")

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Параметры
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Устройство
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"

# Регулярное выражение для валидации кода КТРУ
KTRU_PATTERN = r'\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}'

# Путь к моделям
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)