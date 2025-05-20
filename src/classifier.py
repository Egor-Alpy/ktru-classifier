import os
import re
import json
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List, Optional

from src.config import (
    EMBEDDING_MODEL, LLM_MODEL, MODELS_DIR, DEVICE,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K
)
from src.db import qdrant_db
from src.utils import (
    preprocess_text, extract_ktru_code, format_product_attributes
)
from src.models import Product


class KTRUClassifier:
    """Класс для классификации товаров по кодам КТРУ"""

    def __init__(self):
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.load_models()

    def load_models(self):
        """Загрузка моделей"""
        logger.info("Загрузка моделей")

        # Создание директории для моделей, если она не существует
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Загрузка модели эмбеддингов
        logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE,
            cache_folder=MODELS_DIR
        )

        # Загрузка LLM модели
        logger.info(f"Загрузка LLM модели: {LLM_MODEL}")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            cache_dir=MODELS_DIR
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16,  # Использование half precision для экономии памяти
            device_map="auto",  # Автоматическое размещение на доступных GPU
            cache_dir=MODELS_DIR
        )

        logger.info("Модели успешно загружены")

    def prepare_product_text(self, product: Dict[str, Any]) -> str:
        """Подготовка текстового представления товара"""
        title = product.get("title", "")
        description = product.get("description", "")
        brand = product.get("brand", "")
        category = product.get("category", "")

        # Обработка атрибутов
        attributes_text = format_product_attributes(product.get("attributes", []))

        product_text = f"{title}. {description}. Бренд: {brand}. Категория: {category}. {attributes_text}"
        return preprocess_text(product_text)

    def generate_prompt(self, product: Dict[str, Any], search_results: List[Dict[str, Any]]) -> str:
        """Генерация промпта для LLM"""
        # Подготовка информации о товаре
        product_json = json.dumps(product, ensure_ascii=False, indent=2)

        # Подготовка информации о возможных КТРУ кодах
        context = "Информация о возможных КТРУ кодах:\n\n"
        for hit in search_results:
            ktru_info = hit.payload
            context += f"Код КТРУ: {ktru_info['ktru_code']}\n"
            context += f"Название: {ktru_info['title']}\n"
            context += f"Описание: {ktru_info['description']}\n"

            if "attributes" in ktru_info:
                context += "Характеристики:\n"
                for attr in ktru_info["attributes"]:
                    if "attr_name" in attr:
                        context += f"- {attr['attr_name']}: "
                        if "attr_values" in attr:
                            values = []
                            for val in attr["attr_values"]:
                                if isinstance(val, dict):
                                    value = val.get("value", "")
                                    unit = val.get("value_unit", "")
                                    values.append(f"{value} {unit}".strip())
                            context += ", ".join(values)
                        context += "\n"
            context += "\n---\n\n"

        # Формирование полного промпта
        prompt = f"""Я предоставлю тебе JSON-файл с описанием товара и некоторые возможные коды КТРУ.
Твоя задача - определить единственный точный код КТРУ для этого товара. Если ты не можешь определить код с высокой уверенностью (более 95%), ответь только "код не найден".

## Правила определения:
1. Анализируй все поля предоставленного товара
2. Для корректного определения кода КТРУ обязательно учитывай:
   - Точное соответствие типа товара
   - Технические характеристики
   - Специфические особенности товара

## Информация о товаре:
{product_json}

## Возможные коды КТРУ:
{context}

## Формат ответа:
- Если определен один точный код с уверенностью >95%, выведи только этот код КТРУ
- Если невозможно определить точный код, выведи только фразу "код не найден"

Код КТРУ:"""

        return prompt

    def generate_llm_response(self, prompt: str) -> str:
        """Генерация ответа с помощью LLM"""
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)

        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P
            )

        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлечение только ответа модели (после "Код КТРУ:")
        llm_response = response.split("Код КТРУ:")[-1].strip()
        return llm_response

    def classify(self, product_data: Dict[str, Any]) -> str:
        """Классификация товара по коду КТРУ"""
        try:
            # Проверка и преобразование product_data в объект Product при необходимости
            if not isinstance(product_data, Product):
                product = Product(**product_data)
            else:
                product = product_data

            # Подготовка текстового представления товара
            product_text = self.prepare_product_text(product_data)

            # Создание эмбеддинга для товара
            with torch.no_grad():
                product_embedding = self.embedding_model.encode(product_text, convert_to_tensor=True)
                product_embedding_list = product_embedding.cpu().numpy().tolist()

            # Поиск наиболее похожих КТРУ кодов
            search_results = qdrant_db.search(
                query_vector=product_embedding_list,
                limit=TOP_K
            )

            if not search_results:
                logger.warning("Не найдено подходящих КТРУ кодов")
                return "код не найден"

            # Генерация промпта для LLM
            prompt = self.generate_prompt(product_data, search_results)

            # Генерация ответа с помощью LLM
            llm_response = self.generate_llm_response(prompt)

            # Извлечение кода КТРУ из ответа
            ktru_code = extract_ktru_code(llm_response)

            return ktru_code
        except Exception as e:
            logger.error(f"Ошибка при классификации товара: {e}")
            return "код не найден"