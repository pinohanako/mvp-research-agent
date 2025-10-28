import os
import re

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from agent.src.rag.utils import get_embeddings
from agent.src.utils import logger
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = re.sub(r'[^a-z0-9-]', '-', os.getenv("COLLECTION_NAME").lower()).strip('-')
DIMENSIONS = int(os.getenv("VECTOR_DIMENSION"))
class PdfRetrieval:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)

    async def retrieve(self, query_text: str, pdf_id: str, top_k: int = 10):
        if not pdf_id:
            raise ValueError("pdf_id обязателен для поиска")

        query_emb = await get_embeddings([query_text], task="retrieval.query")
        query_vector = query_emb[0]

        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter={"pdf_id": {"$eq": pdf_id}},
            )

        except Exception as e:
            logger.error(f"Ошибка поиска в Pinecone: {e}")
            return []

        if not results or not getattr(results, "matches", []):
            return []

        matches = results.matches
        logger.info(f"🔎 Найдено {len(matches)} фрагментов для pdf_id={pdf_id}")

        return [m.metadata for m in matches]