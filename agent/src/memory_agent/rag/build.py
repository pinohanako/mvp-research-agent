from .utils import CustomChunker, get_embeddings
from agent.src.memory_agent.utils import logger

import os
import uuid
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()
COLLECTION_NAME = "pdf_2"
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", 1024))

qdrant = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

async def upload_batch(points):
    try:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Батч из {len(points)} точек загружен")
    except Exception as e:
        logger.error(f"Ошибка при загрузке батча в Qdrant: {e}")
        raise


async def process_pdf(file_path, pdf_name, pdf_id, batch_size=64):
    logger.info(f"Начало обработки PDF '{pdf_name}'")

    chunker = CustomChunker()
    chunks_info = chunker.split_text(file_path)
    logger.info(f"PDF разбит на {len(chunks_info)} чанков")

    all_chunks = []
    tasks = []

    for i in range(0, len(chunks_info), batch_size):
        batch_chunks = chunks_info[i:i + batch_size]
        texts = [c["chunk_text"] for c in batch_chunks]

        try:
            embeddings = get_embeddings(texts)
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов: {e}")
            raise

        if len(embeddings) != len(batch_chunks):
            raise ValueError("Количество эмбеддингов не совпадает с количеством чанков!")

        points = []
        for chunk, vector in zip(batch_chunks, embeddings):
            payload = {
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "pdf_file_name": chunk.get("pdf_file_name"),
                "page_number": chunk.get("page_number")
            }
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            ))
            all_chunks.append({
                "chunk_text": chunk["chunk_text"],
                "page_number": chunk.get("page_number"),
                "payload": payload,
                "vector": vector
            })

        tasks.append(asyncio.to_thread(upload_batch, points))

    await asyncio.gather(*tasks)
    logger.info(f"Обработка PDF '{pdf_name}' завершена, всего чанков: {len(all_chunks)}")

    return {
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "file_path": str(file_path),
        "num_chunks": len(all_chunks),
        "chunks": all_chunks
    }