import os
import re
import aiohttp
import spacy
import tiktoken

from semchunk import chunkerify
from spacypdfreader.spacypdfreader import pdf_reader
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

MODEL = "jina-embeddings-v3"
DIMENSIONS = int(os.getenv("VECTOR_DIMENSION"))

async def get_embeddings(texts, task="retrieval.passage"):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "task": task,
        "input": texts
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(JINA_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Jina API error {resp.status}: {text}")
            data = await resp.json()
            return [d["embedding"] for d in data["data"]]

nlp = spacy.load("xx_sent_ud_sm")
class CustomChunker:
    def __init__(self, chunk_size=800, chunk_overlap=200, min_chunk_length=300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.nlp = nlp
        tokenizer = tiktoken.get_encoding("cl100k_base")

        self.chunker = chunkerify(
            tokenizer_or_token_counter=tokenizer,
            chunk_size=self.chunk_size,
            memoize=True,
            cache_maxsize=2048,
        )

    def normalize_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r'(\w)[-–—]\s+(\w)', r'\1\2', text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\s+([.,:;!?])", r"\1", text)
        text = re.sub(r'([.\-–—_•·∙])\1{2,}', r'\1', text)
        text = re.sub(r'\s*[.\-–—_•·∙]{3,}\s*', ' ', text)
        return text.strip()

    def _char_to_page(self, char_pos, page_map):
        for length, page in page_map:
            if char_pos < length:
                return page
        return page_map[-1][1]

    def split_text(self, file_path):
        doc = pdf_reader(file_path, self.nlp)

        all_text = ""
        page_map = []

        for page_num in range(doc._.first_page, doc._.last_page + 1):
            page_text = doc._.page(page_num)
            text_str = getattr(page_text, "text", str(page_text))
            all_text += text_str.strip() + "\n\n"
            page_map.append((len(all_text), page_num))

        normalized_text = self.normalize_text(all_text)

        raw_chunks = self.chunker(
            normalized_text,
            overlap=self.chunk_overlap,
        )

        chunks = []
        for chunk_text in raw_chunks:
            char_pos = all_text.find(chunk_text[:50])
            page_number = self._char_to_page(char_pos, page_map)
            chunks.append({
                "chunk_text": chunk_text.strip(),
                "page_number": page_number,
                "pdf_file_name": file_path,
            })
        return chunks