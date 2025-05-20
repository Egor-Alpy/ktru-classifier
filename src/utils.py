import re
import json
import logging
from typing import Dict, List, Any, Union, Optional
from loguru import logger
from datetime import datetime

from src.config import KTRU_PATTERN


def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ktru_classifier.log')
        ]
    )
    logger.add(
        "logs/ktru_classifier_{time}.log",
        rotation="100 MB",
        retention="30 days",
        level="INFO"
    )


def format_ktru_attributes(attributes: List[Dict[str, Any]]) -> str:
    """Форматирование атрибутов КТРУ в текстовое представление"""
    attributes_text = ""
    if attributes:
        for attr in attributes:
            attr_name = attr.get("attr_name", "")
            if "attr_values" in attr and isinstance(attr["attr_values"], list):
                values = []
                for val in attr["attr_values"]:
                    if isinstance(val, dict):
                        value = val.get("value", "")
                        unit = val.get("value_unit", "")
                        values.append(f"{value} {unit}".strip())
                if values:
                    attributes_text += f"{attr_name}: {', '.join(values)}. "
            elif "attr_value" in attr:
                attributes_text += f"{attr_name}: {attr['attr_value']}. "
    return attributes_text


def format_product_attributes(attributes: List[Dict[str, Any]]) -> str:
    """Форматирование атрибутов товара в текстовое представление"""
    attributes_text = ""
    if attributes:
        for attr in attributes:
            attr_name = attr.get("attr_name", "")
            attr_value = attr.get("attr_value", "")
            if attr_value and attr_value != "Нет данных":
                attributes_text += f"{attr_name}: {attr_value}. "
    return attributes_text


def preprocess_text(text: str) -> str:
    """Предобработка текста"""
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    # Удаление специальных символов
    text = re.sub(r'[^\w\s\.\,\:\;\-\"]', ' ', text)
    # Приведение к нижнему регистру
    text = text.lower()
    return text


def extract_ktru_code(text: str) -> str:
    """Извлечение кода КТРУ из текста"""
    # Поиск кода КТРУ в тексте
    match = re.search(KTRU_PATTERN, text)
    if match:
        return match.group(0)

    # Проверка на "код не найден"
    if "код не найден" in text.lower():
        return "код не найден"

    # Если код не найден, возвращаем "код не найден"
    return "код не найден"


def save_last_sync_time():
    """Сохранение времени последней синхронизации"""
    now = datetime.now().isoformat()
    with open("data/last_sync.json", "w") as f:
        json.dump({"last_sync": now}, f)
    logger.info(f"Время последней синхронизации сохранено: {now}")


def get_last_sync_time() -> Optional[str]:
    """Получение времени последней синхронизации"""
    try:
        with open("data/last_sync.json", "r") as f:
            data = json.load(f)
            return data.get("last_sync")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Не удалось получить время последней синхронизации")
        return None