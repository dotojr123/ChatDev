import uuid
from typing import List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

from chatdev.memory.base import VectorStore, MemoryItem
from chatdev.settings import settings
import structlog

logger = structlog.get_logger()

class QdrantVectorStore(VectorStore):
    def __init__(self):
        # Handle local mode (memory/disk) vs server mode
        if settings.QDRANT_URL == ":memory:" or (settings.QDRANT_URL and not settings.QDRANT_URL.startswith("http")):
             self.client = AsyncQdrantClient(location=settings.QDRANT_URL)
        else:
             self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        self.collection_name = settings.MEMORY_COLLECTION_NAME
        self.vector_size = 1536 # Default for OpenAI embedding, should be configurable

    async def initialize(self):
        """Creates collection if not exists"""
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection", collection=self.collection_name)

    async def add(self, items: List[MemoryItem]) -> None:
        points = []
        for item in items:
            point_id = item.id if item.id else str(uuid.uuid4())
            if not item.vector:
                logger.warning("Skipping item without vector", item_content=item.content[:20])
                continue

            points.append(PointStruct(
                id=point_id,
                vector=item.vector,
                payload={
                    "content": item.content,
                    **item.metadata
                }
            ))

        if points:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info("Upserted points to memory", count=len(points))

    async def search(self, query_vector: List[float], limit: int = 5, filter_criteria: Optional[dict] = None) -> List[MemoryItem]:
        query_filter = None
        if filter_criteria:
            conditions = []
            for key, value in filter_criteria.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                query_filter = Filter(must=conditions)

        # Use query_points instead of search
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter
        )

        memory_items = []
        # query_points returns QueryResponse which has points
        for res in results.points:
            payload = res.payload or {}
            content = payload.pop("content", "")
            memory_items.append(MemoryItem(
                id=str(res.id),
                content=content,
                metadata=payload,
                vector=res.vector if hasattr(res, 'vector') else None
            ))
        return memory_items

    async def delete(self, item_ids: List[str]) -> None:
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=item_ids
        )
        logger.info("Deleted items from memory", count=len(item_ids))
