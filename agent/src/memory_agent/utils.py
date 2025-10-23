import os
import logging
import os
import re
import spacy
from spacypdfreader.spacypdfreader import pdf_reader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_first_page_text(pdf_path: str, nlp=None) -> str:
    if nlp is None:
        nlp = spacy.load("xx_sent_ud_sm")
    doc = pdf_reader(pdf_path, nlp)
    first_page = doc._.page(doc._.first_page)
    return getattr(first_page, "text", str(first_page)).strip()

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "agents.log")
if os.path.exists(log_path):
    os.remove(log_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path)],
)
logger = logging.getLogger(__name__)

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
        self.END_MARKERS = [
            "References", "Список использованной литературы",
            "Литература", "Bibliography"
        ]

    def normalize_text(self, text: str) -> str:
        """
        Минимальная нормализация: убираем неразрывные пробелы и дублирующиеся пробелы,
        но сохраняем переносы и дефисы.
        """
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r'\s{2,}', ' ', text)
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
        buffer_start_pos = 0  # для отслеживания позиции в all_text

        for i, sentence in enumerate(sentences):
            # стоп-маркеры — конец документа
            if any(sentence.startswith(marker) for marker in self.END_MARKERS):
                break

            if buffer:
                buffer += " " + sentence
            else:
                buffer = sentence
                buffer_start_pos = all_text.find(sentence, buffer_start_pos)

            # если буфер достиг минимальной длины или конец текста — режем
            if len(buffer) >= self.min_chunk_length or i == len(sentences) - 1:
                # делим через RecursiveCharacterTextSplitter
                for chunk_text in self.splitter.split_text(buffer):
                    # определяем страницу по позиции символа
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
        """
        Универсальный метод: если text — это путь к PDF, парсим PDF.
        Иначе просто режем обычный текст.
        """
        if os.path.exists(text) and text.endswith(".pdf"):
            return self._split_pdf(text)
        else:
            text = self.normalize_text(text)
            return [{"chunk_text": chunk, "page_number": None, "pdf_file_name": None}
                    for chunk in self.splitter.split_text(text)]


