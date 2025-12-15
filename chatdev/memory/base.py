from typing import Any, List, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class MemoryItem(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    vector: Optional[List[float]] = None
    id: Optional[str] = None

class VectorStore(ABC):
    @abstractmethod
    async def add(self, items: List[MemoryItem]) -> None:
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], limit: int = 5, filter_criteria: Optional[dict] = None) -> List[MemoryItem]:
        pass

    @abstractmethod
    async def delete(self, item_ids: List[str]) -> None:
        pass
