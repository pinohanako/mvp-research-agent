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