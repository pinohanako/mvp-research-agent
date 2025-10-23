import streamlit as st
import traceback
import asyncio
import uuid
from pathlib import Path
from agent.src.memory_agent.context import Context
from agent.src.memory_agent.run import run_agent
from agent.src.memory_agent.rag import build              # process_pdf(pdf_path, pdf_name, pdf_id)
from agent.src.memory_agent.context import Context
from agent.src.memory_agent.utils import logger
from langchain.schema import HumanMessage
from langgraph.runtime import Runtime

from agent.src.memory_agent.utils import extract_first_page_text
from agent.src.memory_agent.run import generate_article_info

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Research Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
runtime_instance = Runtime(context=Context(user_id=st.session_state.session_id))

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "state" not in st.session_state or not isinstance(st.session_state.state, dict):
    st.session_state.state = {
        "articles": [],
        "current_article": None,
        "messages": []
    }

state_obj = st.session_state.state
context = Context(user_id=st.session_state.session_id)

st.sidebar.header("Файлы и анализ")
uploaded_file = st.sidebar.file_uploader("Загрузить PDF", type=["pdf"])

if uploaded_file is not None and all(f['name'] != uploaded_file.name for f in st.session_state.uploaded_files):
    tmp_path = Path("/tmp") / uploaded_file.name

    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        pdf_id = str(uuid.uuid4())

        logger.info(f"Начинаем индексацию PDF '{uploaded_file.name}' в Qdrant")
        result = asyncio.run(build.process_pdf(tmp_path, uploaded_file.name, pdf_id=pdf_id))
        num_chunks = len(result.get('chunks', []))
        logger.info(f"Индексация завершена: {num_chunks} чанков создано")

        # Минимальное состояние статьи
        article_state = {
            "pdf_url": str(tmp_path),
            "pdf_id": pdf_id,
            "pdf_name": uploaded_file.name,
            "info_extracted": False
        }
        async def extract_and_generate(article_state):
                first_page_text = extract_first_page_text(article_state["pdf_url"])
                #runtime_instance = Runtime(context=Context(user_id=st.session_state.session_id, model="название_модели"))
                article_info = await generate_article_info(first_page_text, runtime_instance)
                return article_info
        try: 
            article_info = asyncio.run(extract_and_generate(article_state))
            article_state.update({
                "title": article_info.title,
                "summary": article_info.summary,
                "authors": article_info.authors,
                "published": article_info.published,
                "doi": article_info.doi,
                "info_extracted": True
            })
            logger.info(f"ArticleInfo сгенерирован: {article_info.title}")
        except Exception:
            logger.exception("Ошибка генерации ArticleInfo сразу после загрузки PDF")

        st.session_state.current_article = article_state
        st.session_state.state.setdefault("articles", []).append(article_state)
        st.session_state.state["current_article"] = article_state
        st.session_state.uploaded_files.append({"name": uploaded_file.name, "id": pdf_id})
        logger.info(f"State обновлён: статьи = {len(st.session_state.state['articles'])}")

        st.success(f"PDF '{uploaded_file.name}' загружен и проиндексирован. Можем обсудить его в чате")

    except Exception as e:
        logger.error(f"Ошибка при обработке PDF '{uploaded_file.name}': {e}")
        logger.error(traceback.format_exc())
        st.error(f"Не удалось обработать PDF '{uploaded_file.name}'")

st.sidebar.header("Загруженные файлы")
if st.session_state.uploaded_files:
    for f in st.session_state.uploaded_files:
        st.sidebar.write(f"{f['name']} (id: {f['id']})")
else:
    st.sidebar.warning("Файлы не загружены")

if st.sidebar.button("Start Chat"):
    if st.session_state.state.get("current_article") is None:
        st.warning("Сначала загрузите хотя бы один PDF-файл.")
    else:
        st.session_state.start_chat = True

if st.session_state.start_chat:
    for msg in st.session_state.state.get("messages", []):
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", str(msg))
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Введите запрос")
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
                st.error(f"Ошибка при обработке запроса: {e}")

        new_messages = st.session_state.state.get("messages", [])[prev_length:]
        for msg in new_messages:
            role = getattr(msg, "role", "assistant")
            if role == "tool":
                role = "assistant"
            with st.chat_message(role):
                st.markdown(getattr(msg, "content", str(msg)))

else:
    st.write("Пожалуйста, загрузите файлы и нажмите 'Start Chat', чтобы начать.")

