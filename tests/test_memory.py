import pytest
from chatdev.memory import QdrantVectorStore, MemoryItem

@pytest.mark.asyncio
async def test_qdrant_store_lifecycle(mock_settings):
    store = QdrantVectorStore()
    await store.initialize()

    # Test Add
    item = MemoryItem(content="test memory", metadata={"type": "test"}, vector=[0.1] * 1536)
    await store.add([item])

    # Test Search
    results = await store.search(query_vector=[0.1] * 1536, limit=1)
    assert len(results) == 1
    assert results[0].content == "test memory"

    # Test Delete
    await store.delete([results[0].id])
    results_after_delete = await store.search(query_vector=[0.1] * 1536, limit=1)
    assert len(results_after_delete) == 0
