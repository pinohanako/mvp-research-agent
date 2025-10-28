import os
import re
import asyncio
import pytest
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from agent.src.utils import logger

load_dotenv()

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    logger.info("--- Проверка переменных окружения Pinecone ---")
    assert os.getenv("PINECONE_API_KEY"), "Не задан PINECONE_API_KEY"
    assert os.getenv("COLLECTION_NAME"), "Не задано имя индекса Pinecone"
    assert os.getenv("VECTOR_DIMENSION"), "Не задана VECTOR_DIMENSION"
    logger.info("Окружение успешно загружено и проверено.")

@pytest.fixture(scope="session")
def event_loop():
    # отдельный event loop для pytest-asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = re.sub(r'[^a-z0-9-]', '-', os.getenv("COLLECTION_NAME").lower()).strip('-')
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    logger.info(f"Подключение к Pinecone Index '{index_name}' установлено.")
    return index

@pytest.fixture(scope="session")
def test_pdf():
    return {
        "pdf_path": "/Users/pino/Projects/work/sber/mvp/uploaded_files/Введение в травматологию.pdf",
        "pdf_id": "debug-001",
        "pdf_name": "example"
    }