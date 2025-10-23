import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

load_dotenv()
COLLECTION_NAME = "pdf_2"
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", 1024))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qdrant = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

def init_collection():
    """
    Создание коллекции в Qdrant с HNSW индексом для векторов
    pdf_name и page_number будут храниться вместе с точками, но индексироваться не будут
    Из корня проекта запустить: python3 ./agent/src/rag/init_db.py
    """
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        logger.info(f"Коллекция '{COLLECTION_NAME}' уже существует. Удаляем и создаём заново...")
        qdrant.delete_collection(collection_name=COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "embedding": VectorParams(
                size=DIMENSIONS,
                distance=Distance.COSINE,
            ),
        },
        shard_number=1,
        replication_factor=1,
        write_consistency_factor=1,
        on_disk_payload=False,
    )

    logger.info(f"Коллекция '{COLLECTION_NAME}' создана с HNSW индексом для векторов.")

    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="pdf_id",
        field_schema="keyword"
    )
    logger.info("Payload index создан для pdf_id.")

if __name__ == "__main__":
    init_collection()