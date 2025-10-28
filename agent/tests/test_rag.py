import os
import asyncio
import pytest
import numpy as np
import logging
from time import sleep

from agent.src.rag.utils import CustomChunker, get_embeddings
from agent.src.rag.build import process_pdf
from agent.src.rag.service import PdfRetrieval
from .conftest import pinecone_index, test_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_full_rag_pipeline(pinecone_index, test_pdf):
    """
    - разбиение PDF на чанки
    - генерация эмбеддингов
    - загрузку векторов
    - поиск релевантных фрагментов
    """
    pdf_path = test_pdf["pdf_path"]
    pdf_id = test_pdf["pdf_id"]
    pdf_name = test_pdf["pdf_name"]

    logger.info("--- Проверка чанков PDF ---")
    chunker = CustomChunker(chunk_size=900, chunk_overlap=200)
    chunks = chunker.split_text(pdf_path)
    assert chunks, "Не удалось получить чанки из PDF"
    logger.info(f"Получено {len(chunks)} чанков")

    logger.info("--- Проверка получения эмбеддингов ---")
    texts = [c["chunk_text"] for c in chunks[:5]]
    embs = await get_embeddings(texts)
    assert len(embs) == len(texts), "Количество эмбеддингов не совпадает"
    logger.info(f"Эмбеддинги получены, размерность: {len(embs[0])}")

    logger.info("--- Проверка process_pdf и загрузки в Pinecone ---")
    result = await process_pdf(pdf_path, pdf_name=pdf_name, pdf_id=pdf_id)
    assert result["num_chunks"] > 0, "Индексация не выполнена"
    logger.info(f"Загружено {result['num_chunks']} чанков в Pinecone")

    logger.info("Ожидание синхронизации с Pinecone...")
    await asyncio.sleep(5)

    logger.info("--- Проверка поиска в Pinecone ---")
    retriever = PdfRetrieval()
    query = "Какие бывают типы переломов костей?"

    matches = []
    for attempt in range(3):
        matches = await retriever.retrieve(query_text=query, pdf_id=pdf_id, top_k=5)
        if matches:
            logger.info(f"Найдено {len(matches)} релевантных фрагментов (попытка {attempt+1})")
            break
        else:
            logger.warning(f"Результаты пустые, повтор через 3 секунды (попытка {attempt+1}/3)")
            await asyncio.sleep(3)

    assert matches, "Результаты поиска остались пустыми после 3 попыток"

    for i, m in enumerate(matches[:3]):
        snippet = m.get("chunk_text", "")[:120].replace("\n", " ")
        page = m.get("page_number", "?")
        logger.info(f"[{i+1}] страница {page}: {snippet}")

    logger.info("--- Проверка статистики векторов ---")
    arr = np.stack([np.array(v[1]) for v in result["chunks"]])
    norms = np.linalg.norm(arr, axis=1)
    assert not np.isnan(arr).any(), "Найдены NaN в эмбеддингах"
    assert (norms > 0).all(), "Найдены нулевые векторы"
    logger.info("Эмбеддинги корректны, пайплайн работает стабильно")

    logger.info("--- Полный тест RAG-пайплайна для Pinecone завершён успешно ---")


