import os
from qdrant_client import QdrantClient, models
from agent.src.rag.utils import get_embeddings
from dotenv import load_dotenv

load_dotenv()

QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

qdrant = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY, timeout=60.0)

async def retrieve_chunks(query_text: str, pdf_id: str, top_k: int = 10):
    query_emb = await get_embeddings([query_text], task="retrieval.query")
    query_vector = query_emb[0]

    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="pdf_id",
                    match=models.MatchValue(value=pdf_id)
                )]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при поиске в Qdrant: {e}")

    if not results or not results.points:
        return []

    return [p.payload for p in results.points]
