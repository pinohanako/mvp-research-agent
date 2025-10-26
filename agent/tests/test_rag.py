import os
import pytest
import numpy as np
from qdrant_client import models
from agent.src.utils import logger
from agent.src.rag.utils import CustomChunker, get_embeddings
from agent.src.rag.build import process_pdf
from .conftest import qdrant_client, test_pdf

@pytest.mark.asyncio
async def test_full_rag_pipeline(qdrant_client, test_pdf):
    pdf_path = test_pdf["pdf_path"]
    pdf_id = test_pdf["pdf_id"]
    pdf_name = test_pdf["pdf_name"]

    logger.info("--- Проверка чанков ---")
    chunker = CustomChunker(chunk_size=900, chunk_overlap=200)
    chunks = chunker.split_text(pdf_path)
    assert chunks, "Не удалось получить чанки"
    logger.info(f"Получено {len(chunks)} чанков")

    logger.info("--- Проверка eмбеддингов ---")
    texts = [c["chunk_text"] for c in chunks[:5]]
    embs = await get_embeddings(texts)
    assert len(embs) == len(texts), "Количество эмбеддингов не совпадает"
    logger.info(f"Эмбеддинги получены. Размерность: {len(embs[0])}")

    logger.info("--- Проверка process_pdf ---")
    result = await process_pdf(pdf_path, pdf_name=pdf_name, pdf_id=pdf_id, batch_size=20)
    assert result["num_chunks"] > 0, "Индексирование не выполнено"
    logger.info(f"Индексировано {result['num_chunks']} чанков в Qdrant")

    logger.info("--- Проверка поиска ---")
    query = "Какие бывают типы переломов костей?"
    query_emb = await get_embeddings([query], task="retrieval.query")
    query_vector = query_emb[0]

    results = qdrant_client.query_points(
        collection_name=os.getenv("COLLECTION_NAME"),
        query=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="pdf_id",
                match=models.MatchValue(value=pdf_id)
            )]
        ),
        limit=5,
        with_payload=True
    )
    assert results.points, "Результаты поиска пустые"
    logger.info(f"Найдено {len(results.points)} совпадений в Qdrant")

    for p in results.points[:3]:
        page = p.payload.get("page_number", "?")
        snippet = p.payload.get("chunk_text", "")[:120].replace("\n", " ")
        logger.debug(f"[стр. {page}] {snippet}")

    logger.info("--- Проверка статистики эмбеддингов ---")
    arr = np.stack([np.array(c["vector"]) for c in result["chunks"]])
    norms = np.linalg.norm(arr, axis=1)
    assert not np.isnan(arr).any(), "В эмбеддингах найдены NaN"
    logger.info("--- Полный тест RAG-пайплайна завершён успешно ---")
