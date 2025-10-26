from agent.src.rag.utils import CustomChunker, get_embeddings
from agent.src.utils import logger

import os
import uuid
import asyncio
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Batch

load_dotenv()

QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DIMENSIONS = int(os.getenv("VECTOR_DIMENSION"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

qdrant = AsyncQdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY, timeout=10.0)

SEMAPHORE = asyncio.Semaphore(20)

async def upload_batch(points_batch: Batch):
    async with SEMAPHORE:
        try:
            await qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch,
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке батча: {e}")
            raise

async def process_pdf(file_path, pdf_name, pdf_id, batch_size=50):
    chunker = CustomChunker()
    chunks_info = chunker.split_text(file_path)
    total_chunks = len(chunks_info)

    all_chunks = []
    upload_tasks = []

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks_info[i:i + batch_size]
        texts = [c["chunk_text"] for c in batch_chunks]

        embeddings = await get_embeddings(texts)

        batch = Batch(
            ids=[str(uuid.uuid4()) for _ in embeddings],
            vectors=embeddings,
            payloads=[
                {
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "pdf_file_name": c.get("pdf_file_name"),
                    "page_number": c.get("page_number"),
                    "chunk_text": c["chunk_text"],
                }
                for c in batch_chunks
            ],
        )

        upload_tasks.append(upload_batch(batch))

        all_chunks.extend([
            {
                "payload": batch.payloads[idx],
                "vector": embeddings[idx],
            }
            for idx in range(len(batch_chunks))
        ])

    await asyncio.gather(*upload_tasks)
    logger.info(f"Количество запушенных чанков: {len(all_chunks)}")

    return {
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "file_path": str(file_path),
        "num_chunks": len(all_chunks),
        "chunks": all_chunks,
    }