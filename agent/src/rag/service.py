import os
from .utils import get_embeddings
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from agent.src.rag.build import qdrant
#from agent.src.rag import get_embeddings

load_dotenv()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

async def fetch_relevant_chunks(pdf_id: str, query_text: str, top_k: int = 5):
    """
    Получаем embedding запроса, ищем топ-N релевантных чанков в Qdrant
    Возвращает список текстов чанков
    """
    query_vector = get_embeddings([query_text], task="retrieval.query")[0]

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="pdf_id",
                match=models.MatchValue(value=pdf_id)
            )]
        ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=top_k
    )

    return [p.payload["document"] for p in results.result]