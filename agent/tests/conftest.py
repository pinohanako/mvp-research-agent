import os
import asyncio
import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from agent.src.utils import logger

load_dotenv()

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    logger.info("--- Загрузка .env и проверка переменных окружения ---")
    assert os.getenv("QDRANT_ENDPOINT"), "Не задан QDRANT_ENDPOINT"
    assert os.getenv("QDRANT_API_KEY"), "Не задан QDRANT_API_KEY"
    assert os.getenv("COLLECTION_NAME"), "Не задано имя коллекции Qdrant"
    logger.info("Окружение успешно загружено.")

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def qdrant_client():
    client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60.0
    )
    return client

@pytest.fixture(scope="session")
def test_pdf():
    return {
        "pdf_path": "/Users/pino/Projects/work/sber/mvp/uploaded_files/Введение в травматологию.pdf",
        "pdf_id": "debug-001",
        "pdf_name": "example"
    }