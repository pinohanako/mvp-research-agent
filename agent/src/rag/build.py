import os
import re
import uuid
import asyncio
import itertools
import random
import time
from dotenv import load_dotenv

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from agent.src.rag.utils import (CustomChunker, 
                                 get_embeddings, 
                                 normalize_metadata, 
                                 batch_iterable)
from agent.src.utils import logger

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = re.sub(r'[^a-z0-9-]', '-', os.getenv("COLLECTION_NAME").lower()).strip('-')
DIMENSIONS = int(os.getenv("VECTOR_DIMENSION"))

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)
logger.info(f"-- Подключились к Pinecone Index '{INDEX_NAME}'")

MAX_PARALLEL = 10
MAX_RETRIES = 3
BATCH_SIZE = 250
SEMAPHORE = asyncio.Semaphore(MAX_PARALLEL)

async def upload_batch(batch, attempt=1):
    async with SEMAPHORE:
        try:
            async_result = index.upsert(vectors=batch, async_req=True)
            async_result.result()
        except Exception as e:
            if attempt <= MAX_RETRIES:
                wait_time = 2 ** attempt + random.random()
                logger.warning(
                    f"Ошибка upsert (попытка {attempt}/{MAX_RETRIES}), retry через {wait_time:.1f}s: {e}"
                )
                await asyncio.sleep(wait_time)
                await upload_batch(batch, attempt + 1)
            else:
                logger.error(f"Ошибка после {MAX_RETRIES} попыток: {e}")
                raise

async def process_pdf(file_path, pdf_name, pdf_id):
    chunker_util = CustomChunker()
    chunks_info = chunker_util.split_text(file_path)
    total_chunks = len(chunks_info)
    logger.info(f"'{pdf_name}' --- {total_chunks} чанков")

    async def generate_vectors():
        texts = [c["chunk_text"] for c in chunks_info]
        embeddings = await get_embeddings(texts)

        for idx, c in enumerate(chunks_info):
            metadata = {
                "pdf_id": str(pdf_id),
                "pdf_name": str(pdf_name),
                "page_number": int(c.get("page_number")),
                "chunk_text": str(c["chunk_text"]),
            }
            yield (
                str(uuid.uuid4()),
                embeddings[idx],
                normalize_metadata(metadata),
            )

    all_vectors = [v async for v in generate_vectors()]

    start = time.time()
    async_results = [
        asyncio.create_task(upload_batch(batch))
        for batch in batch_iterable(all_vectors, batch_size=BATCH_SIZE)
    ]
    await asyncio.gather(*async_results)

    elapsed = time.time() - start
    logger.info(
        f"Загружено за {elapsed:.2f}с (~{len(all_vectors)/elapsed:.1f} vectors/s)"
    )

    return {
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "file_path": str(file_path),
        "num_chunks": len(all_vectors),
        "chunks": all_vectors,
    }