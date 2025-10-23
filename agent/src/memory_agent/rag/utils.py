import os
import re
import requests

import spacy
from spacypdfreader.spacypdfreader import pdf_reader
from langchain_text_splitters import RecursiveCharacterTextSplitter

JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

MODEL = "jina-embeddings-v3"
DIMENSIONS = 1024

def get_embeddings(texts, task="retrieval.passage"):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "task": task,
        "input": texts
    }
    resp = requests.post(JINA_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text}")
    data = resp.json()
    return [d["embedding"] for d in data["data"]]

'''
Текст сначала собирается и базово нормализуется.
Потом он делится на предложения.
Из предложений формируются буферы → чанки.
Каждый чанк проходит окончательную нормализацию, вычисляется страница, сохраняется имя файла.
'''
class CustomChunker:
    def __init__(self, chunk_size=600, chunk_overlap=50, min_chunk_length=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.nlp = spacy.load("xx_sent_ud_sm")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def normalize_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _split_pdf(self, file_path):
        doc = pdf_reader(file_path, self.nlp)
        all_text = ""
        page_map = []

        for page_num in range(doc._.first_page, doc._.last_page + 1):
            page_text = doc._.page(page_num)
            text_str = getattr(page_text, "text", str(page_text))
            all_text += text_str + "\n\n"
            page_map.append((len(all_text), page_num))

        spacy_doc = self.nlp(all_text)
        sentences = [s.text.strip() for s in spacy_doc.sents if s.text.strip()]

        chunks = []
        buffer = ""
        buffer_start_pos = 0

        for i, sentence in enumerate(sentences):
            if buffer:
                buffer += " " + sentence
            else:
                buffer = sentence
                buffer_start_pos = all_text.find(sentence, buffer_start_pos)

            if len(buffer) >= self.min_chunk_length or i == len(sentences) - 1:
                for chunk_text in self.splitter.split_text(buffer):
                    char_pos = all_text.find(chunk_text, buffer_start_pos)
                    page_number = self._char_to_page(char_pos, page_map)
                    chunks.append({
                        "chunk_text": self.normalize_text(chunk_text),
                        "page_number": page_number,
                        "pdf_file_name": file_path
                    })
                buffer = ""

        return chunks

    def _char_to_page(self, char_pos, page_map):
        for length, page in page_map:
            if char_pos < length:
                return page
        return page_map[-1][1]

    def split_text(self, text: str):
        if os.path.exists(text):
            return self._split_pdf(text)
        else:
            text = self.normalize_text(text)
            return [
                {"chunk_text": chunk,     # сам текст чанка
                 "page_number": None,     # номер страницы, вычисленный через _char_to_page
                 "pdf_file_name": None}   # название PDF-файла
                for chunk in self.splitter.split_text(text)
            ]