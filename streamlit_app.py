from agent.src.context import Context
from agent.src.run import run_agent, generate_article_info
from agent.src.rag import build
from agent.src.utils import logger, extract_first_page_text

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

import streamlit as st
import asyncio
import uuid
import traceback
from pathlib import Path

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Research Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "state" not in st.session_state or not isinstance(st.session_state.state, dict):
    st.session_state.state = {
        "articles": [],
        "current_article": None,
        "messages": [],
        "analysis_mode": False,
        "pdf_id": None,
        "retrieved_chunks": [],
    }

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

runtime_instance = Runtime(context=Context(user_id=st.session_state.session_id))

st.sidebar.header("Файлы и анализ")
uploaded_file = st.sidebar.file_uploader("Загрузить PDF", type=["pdf"])

if uploaded_file is not None and all(f["name"] != uploaded_file.name for f in st.session_state.uploaded_files):
    tmp_path = Path("/tmp") / uploaded_file.name

    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        pdf_id = str(uuid.uuid4())
        result = asyncio.run(build.process_pdf(tmp_path, uploaded_file.name, pdf_id))
        logger.info(f"🦔 Количество полученных фрагментов: {result['num_chunks']} ")

        def extract_first_page_info():
            first_page_text = extract_first_page_text(tmp_path)
            return asyncio.run(generate_article_info(first_page_text, runtime_instance))

        try:
            article_info = extract_first_page_info()
        except Exception:
            article_info = None
            logger.exception("Ошибка генерации ArticleInfo после загрузки PDF")

        article_state = {
            "pdf_id": pdf_id,
            "pdf_name": uploaded_file.name,
            "title": getattr(article_info, "title", None),
            "summary": getattr(article_info, "summary", None),
            "authors": getattr(article_info, "authors", None),
            "published": getattr(article_info, "published", None),
            "doi": getattr(article_info, "doi", None),
        }

        st.session_state.uploaded_files.append({"name": uploaded_file.name, "id": pdf_id})
        st.session_state.state["current_article"] = article_state
        st.session_state.state["articles"].append(article_state)
        st.session_state.state["analysis_mode"] = True
        st.session_state.state["pdf_id"] = pdf_id
        st.session_state.state["retrieved_chunks"] = []

        st.success(f"PDF '{uploaded_file.name}' загружен и проиндексирован. Вы вошли в режим анализа.")

    except Exception as e:
        logger.error(f"Ошибка при загрузке PDF '{uploaded_file.name}': {e}")
        logger.error(traceback.format_exc())
        st.error(f"Не удалось обработать PDF '{uploaded_file.name}'")

st.sidebar.header("Загруженные файлы")
if st.session_state.uploaded_files:
    for f in st.session_state.uploaded_files:
        st.sidebar.write(f"{f['name']} (id: {f['id']})")
else:
    st.sidebar.info("Файлы пока не загружены")

if st.session_state.uploaded_files:
    if st.session_state.state.get("analysis_mode", False):
        st.sidebar.markdown("Сейчас вы в режиме анализа PDF")
        if st.sidebar.button("Выйти из режима анализа"):
            st.session_state.state["analysis_mode"] = False
            st.session_state.state["current_article"] = None
            st.session_state.state["pdf_id"] = None
            st.session_state.state["retrieved_chunks"] = []
            st.sidebar.success("Режим анализа завершён. Теперь вы в обычном чате.")
    else:
        st.sidebar.info("Вы в режиме чата")
        if st.session_state.state.get("articles"):
            for art in st.session_state.state["articles"]:
                if st.sidebar.button(f"Перейти к анализу: {art.get('title') or art['pdf_name']}"):
                    st.session_state.state["current_article"] = art
                    st.session_state.state["analysis_mode"] = True
                    st.session_state.state["pdf_id"] = art["pdf_id"]
                    st.sidebar.success(f"Вы перешли к анализу '{art.get('title') or art['pdf_name']}'")

for msg in st.session_state.state.get("messages", []):
    role = getattr(msg, "role", "user")
    content = getattr(msg, "content", str(msg))
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Введите ваш запрос")
if user_input:
    user_msg = HumanMessage(content=user_input)
    st.session_state.state.setdefault("messages", []).append(user_msg)

    with st.chat_message("user"):
        st.markdown(user_input)

    prev_length = len(st.session_state.state["messages"])

    with st.spinner("Обрабатываем запрос..."):
        try:
            result = asyncio.run(
                run_agent(
                    user_input,
                    st.session_state.state,
                    st.session_state.session_id
                )
            )
            st.session_state.state = result
        except Exception as e:
            logger.exception("Ошибка при обработке запроса")
            st.error(f"Ошибка при обработке запроса: {e}")

    new_messages = st.session_state.state.get("messages", [])[prev_length:]
    for msg in new_messages:
        role = getattr(msg, "role", "assistant")
        with st.chat_message(role):
            st.markdown(getattr(msg, "content", str(msg)))