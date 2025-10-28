import os
import re
import logging
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = re.sub(r'[^a-z0-9-]', '-', os.getenv("COLLECTION_NAME").lower()).strip('-')
DIMENSIONS = int(os.getenv("VECTOR_DIMENSION"))

CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_index():
    """
    Создаем Pinecone Index, если он не существует
    Из корня проекта запустить:
        python3 ./agent/src/rag/__init__.py
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes().indexes]

    if INDEX_NAME in existing_indexes:
        logger.info(f"Индекс '{INDEX_NAME}' уже существует, пропускаем создание.")
        return

    logger.info(f"Создаём новый индекс: '{INDEX_NAME}' (dim={DIMENSIONS})")

    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )

    logger.info(f"-- Индекс '{INDEX_NAME}' успешно создан.")

if __name__ == "__main__":
    init_index()