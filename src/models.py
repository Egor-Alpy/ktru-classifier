from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime

from src.config import KTRU_PATTERN


class Attribute(BaseModel):
    """Модель атрибута товара"""
    attr_name: str
    attr_value: Optional[str] = None
    attr_values: Optional[List[Dict[str, str]]] = None


class Supplier(BaseModel):
    """Модель поставщика товара"""
    supplier_name: str
    supplier_tel: Optional[str] = None
    supplier_address: Optional[str] = None
    supplier_description: Optional[str] = None
    supplier_offers: Optional[List[Dict[str, Any]]] = None


class Product(BaseModel):
    """Модель товара"""
    title: str
    description: Optional[str] = ""
    article: Optional[str] = None
    brand: Optional[str] = None
    country_of_origin: Optional[str] = None
    warranty_months: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    attributes: Optional[List[Attribute]] = []
    suppliers: Optional[List[Supplier]] = []


class KTRUItem(BaseModel):
    """Модель КТРУ кода"""
    ktru_code: str
    title: str
    keywords: Optional[List[str]] = []
    description: Optional[str] = ""
    unit: Optional[str] = None
    version: Optional[str] = None
    updated_at: Optional[str] = None
    source_link: Optional[str] = None
    attributes: Optional[List[Attribute]] = []

    @field_validator('ktru_code')
    def validate_ktru_code(cls, v):
        if not re.match(KTRU_PATTERN, v):
            raise ValueError(f"Неверный формат кода КТРУ: {v}")
        return v


class ProductClassificationRequest(BaseModel):
    """Запрос на классификацию товара"""
    product: Product


class ProductClassificationResponse(BaseModel):
    """Ответ с кодом КТРУ для товара"""
    ktru_code: str

    @field_validator('ktru_code')
    def validate_ktru_code(cls, v):
        if v != "код не найден" and not re.match(KTRU_PATTERN, v):
            raise ValueError(f"Неверный формат кода КТРУ: {v}")
        return v


class SearchResult(BaseModel):
    """Результат поиска в векторной базе данных"""
    id: int
    score: float
    payload: Dict[str, Any]