from agent.src.state import State
from agent.src.context import Context
from agent.src.prompts import Intent, Filter, ArticleInfo, PdfContextDecision
from agent.src.tracing import traced
from agent.src import tools
from agent.src.rag import service
from agent.src.utils import logger

from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.cache.memory import InMemoryCache
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore, TTLConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import os
import asyncio
import aiohttp
import feedparser
from datetime import datetime
from typing import cast
from collections import defaultdict
from dotenv import load_dotenv

DB_URI = f"postgresql://{os.environ['POSTGRES_USER']}:" \
         f"{os.environ['POSTGRES_PASSWORD']}@" \
         f"{os.environ['POSTGRES_HOST']}:" \
         f"{os.environ['POSTGRES_PORT']}/" \
         f"{os.environ['POSTGRES_DB']}?sslmode=disable&connect_timeout=10"

load_dotenv()

@traced
async def call_model(state: dict, runtime: Runtime[Context]) -> dict:
    logger.info("🦔 ENTER NODE: call_model")

    user_id = runtime.context.user_id
    model_str = runtime.context.model
    system_prompt_template = runtime.context.system_prompt
    model_name = f"accounts/fireworks/models/{model_str.split('/')[-1]}"

    state.setdefault("intent", "qa")
    last_user_message = state["messages"][-1].content if state.get("messages") else ""

    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state["messages"][-3:]]),
        limit=10,
    )
    formatted_memories = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted_memories:
        formatted_memories = f"<memories>\n{formatted_memories}\n</memories>"

    try:
        sys_prompt = system_prompt_template.format_map(defaultdict(str, {
            "user_info": formatted_memories or "",
            "time": datetime.now().isoformat()
        }))
    except Exception:
        sys_prompt = str(system_prompt_template)
        logger.exception("System prompt formatting failed; using raw template")

    client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1"
    )

    messages_payload = [{"role": "system", "content": sys_prompt}]
    for msg in state["messages"]:
        messages_payload.append({
            "role": getattr(msg, "role", "user"),
            "content": getattr(msg, "content", str(msg))
        })

    pdf_id = state.get("current_article", {}).get("pdf_id")
    if pdf_id:
        logger.info(f"PDF ID найден в state: {pdf_id}, проверяем, актуален ли он для продолжения диалога")

        parser = PydanticOutputParser(pydantic_object=PdfContextDecision)
        title = state.get("title")
        pdf_context_prompt = PromptTemplate(
            template=(
                "Определи, продолжает ли пользователь обсуждать PDF '{title}'"
                "Ответь строго в JSON формате.\n\n"
                "Требуемая схема:\n{format_instructions}\n\n"
                "Сообщение пользователя:\n{query}"
            ),
            input_variables=["query", "title"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user", 
                    "content": pdf_context_prompt.format(query=last_user_message, title=title)}],
                max_tokens=500,
                temperature=0.0,
            )
            parsed = parser.parse(response.choices[0].message.content)
            raw_value = getattr(parsed, "continue_pdf", False)
            if isinstance(raw_value, bool):
                continue_pdf = raw_value
            else:
                continue_pdf = str(raw_value).strip().lower() == "true"
        except Exception as e:
            logger.warning(f"!!! Ошибка при StructuredOutput проверки PDF-контекста: {e}")
            continue_pdf = False

        if continue_pdf:
            logger.info("🦔 Модель подтвердила, что пользователь продолжает обсуждать PDF.")
            state["intent"] = "analyze"
            state["query"] = last_user_message

            try:
                retriever = service.PdfRetrieval()
                chunks = await retriever.retrieve(last_user_message, pdf_id, top_k=8)
                state["retrieved_chunks"] = chunks or []
                logger.info(f"🦔 Извлечено {len(chunks)} чанков из Qdrant для pdf_id={pdf_id}")
            except Exception as e:
                logger.exception(f"Ошибка при поиске чанков: {e}")
                state["retrieved_chunks"] = []

            return state

        else:
            logger.info("🦔 Модель решила, что пользователь сменил тему — очищаем PDF контекст")
            state["pdf_id"] = None
            state["retrieved_chunks"] = []

    # --- Классификация намерения (qa или search)
    parser = PydanticOutputParser(pydantic_object=Intent)
    intent_prompt_template = PromptTemplate(
        template=(
            "Классифицируй намерение пользователя на одну из категорий: qa, search.\n"
            "Если запрос явно относится к поиску статей — 'search'.\n"
            "Если это вопрос, свободный текст, комментарий, приветствие — 'qa'.\n\n"
            "Верни ТОЛЬКО JSON, соответствующий схеме:\n{format_instructions}\n\n"
            "Запрос пользователя:\n{query}"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    try:
        intent_response = client.chat.completions.create(
            model=model_name,
            messages=messages_payload + [
                {"role": "user", "content": intent_prompt_template.format(query=last_user_message)}
            ],
            max_tokens=50,
            temperature=0.0,
        )
        intent_parsed = parser.parse(intent_response.choices[0].message.content)
        intent_str = intent_parsed.intent
    except Exception as e:
        logger.warning(f"!!! Ошибка определения intent: {e}")
        intent_str = "qa"

    state["intent"] = intent_str
    state["query"] = last_user_message
    logger.info(f"🧩 Intent определён: {intent_str}")

    if intent_str == "search":
        parser = PydanticOutputParser(pydantic_object=Filter)
        prompt_text = (
            f"Составь JSON по запросу:\n{last_user_message}\n\n"
            f"Строго придерживайся схемы:\n{parser.get_format_instructions()}"
        )
        try:
            llm_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Ты должен вернуть только JSON, без комментариев."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=500,
                temperature=0.0,
            )
            response_json = llm_response.choices[0].message.content
            article_info = parser.parse(response_json)

            state["title"] = article_info.title
            state["authors"] = [article_info.author] if article_info.author else []
            state["doi"] = article_info.doi
            state["current_article"] = {
                "title": state["title"],
                "authors": state["authors"],
                "doi": state["doi"],
            }
        except Exception:
            logger.exception("Ошибка при генерации фильтров поиска")
            state["current_article"] = None

        return state

    try:
        qa_response = client.chat.completions.create(
            model=model_name,
            messages=messages_payload,
            max_tokens=4000,
            temperature=0.5,
        )
        reply_content = qa_response.choices[0].message.content
        state["messages"].append(AIMessage(content=reply_content))
    except Exception as e:
        logger.exception("QA call failed: %s", e)
        state["messages"].append(AIMessage(content="Произошла ошибка при обработке запроса."))

    try:
        await tools.upsert_memory(
            content=last_user_message,
            context=f"Ответ создан {datetime.now().isoformat()}",
            user_id=user_id,
            store=runtime.store
        )
    except Exception:
        logger.exception("upsert_memory failed")

    logger.info(f"call_model завершён: intent={state['intent']}")
    return state

async def store_memory(state: dict, runtime: Runtime[Context]):
    logger.info("🦔 ENTER NODE: store_memory")
    if not state.get("messages"):
        return state

    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None)

    if not tool_calls:
        return state

    saved = await asyncio.gather(
        *(
            tools.upsert_memory(
                **tc["args"],
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store)
            )
            for tc in tool_calls
        )
    )
    for tc, mem in zip(tool_calls, saved):
        state["messages"].append(HumanMessage(content=mem, role="tool"))

    logger.info(f"🦔 store_memory detected intent={state.get('intent')}")
    return state

async def generate_article_info(first_page_text: str, runtime: Runtime[Context]) -> ArticleInfo:
    client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1"
    )

    model_name = f"accounts/fireworks/models/{runtime.context.model.split('/')[-1]}"

    parser = PydanticOutputParser(pydantic_object=ArticleInfo)
    prompt_template = PromptTemplate(
        template=(
            "Извлеки из текста структурированную информацию о статье и верни строго JSON, соответствующий схеме:\n"
            "{format_instructions}\n\n"
            "Текст первой страницы:\n{first_page_text}"
        ),
        input_variables=["first_page_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Используй только предоставленный контекст для извлечения метаданных."},
            {"role": "user", "content": prompt_template.format(first_page_text=first_page_text)}
        ],
        max_tokens=1000,
        temperature=0.0,
    )

    raw_json = response.choices[0].message.content
    article_info = parser.parse(raw_json)
    return article_info

@traced
async def analyze_node(state: dict, runtime: Runtime[Context], top_k: int = 8):
    pdf_id = state.get("pdf_id")
    query_text = state.get("query", "")
    chunks_data = state.get("retrieved_chunks", [])
    current_article = state.get("current_article", {})

    if not chunks_data:
        state["messages"].append(
            AIMessage(
                content=(
                    "Не удалось найти релевантные фрагменты для анализа. "
                    "Попробуй переформулировать вопрос или уточнить контекст."
                )
            )
        )
        return state

    formatted_chunks = []
    for c in chunks_data:
        text = c.get("chunk_text", "").strip()
        page = c.get("page_number", "?")
        if text:
            formatted_chunks.append(f"[стр. {page}] {text}")

    chunks_text = "\n\n".join(formatted_chunks)

    article_info = (
        f"Название статьи: {current_article.get('title') or 'неизвестно'}\n"
        f"Авторы: {', '.join(current_article.get('authors', [])) or 'не указаны'}\n"
        f"DOI: {current_article.get('doi') or 'не нашлось'}\n"
        f"PDF: {current_article.get('pdf_name')}"
    )

    prompt = (
        "Ты — исследовательский ассистент, анализирующий загруженный PDF.\n"
        "Используй только предоставленные фрагменты для ответа.\n"
        "Если цитируешь — указывай страницу в виде [стр. N].\n\n"
        f"Информация о статье:\n{article_info}\n\n"
        f"Фрагменты из PDF:\n{chunks_text}\n\n"
        f"Вопрос пользователя:\n{query_text}"
    )

    client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1"
    )

    model_name = f"accounts/fireworks/models/{runtime.context.model.split('/')[-1]}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Отвечай строго по приведённым фрагментам PDF."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.4,
        )

        answer = response.choices[0].message.content
        logger.info(f"✅ analyze_node: ответ сгенерирован успешно для pdf_id={pdf_id}")
        state["messages"].append(AIMessage(content=answer))

    except Exception as e:
        logger.exception(f"Ошибка вызова модели в analyze_node: {e}")
        state["messages"].append(
            AIMessage(content="Ошибка при генерации ответа на основе PDF")
        )

    return state

MAX_RESULTS_PER_QUERY = 5
@traced
async def arxiv_research(state: dict, runtime: Runtime[Context]):
    query = state.get("query", "")
    if not query:
        return state

    search_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={MAX_RESULTS_PER_QUERY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url) as resp:
            text = await resp.text()
            feed = feedparser.parse(text)

            state.setdefault("raw_results", [])
            state.setdefault("url_sources", [])

            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                pdf_link = next((l.href for l in entry.get("links", []) if l.type == "application/pdf"), None)
                if not pdf_link or any(d["url"] == pdf_link for d in state["raw_results"]):
                    continue

                state["raw_results"].append({
                    "url": pdf_link,
                    "text": f"{title}\n{summary}"
                })
                if pdf_link not in state["url_sources"]:
                    state["url_sources"].append(pdf_link)

    state.setdefault("messages", []).append(
        HumanMessage(
            content=f"arXiv найдено {len(feed.entries)} статей, уникальных URL: {len(state['url_sources'])}",
            role="tool"
        )
    )
    return state

@traced
async def summarize_sources(state: dict, runtime: Runtime[Context]):
    raw_results = state.get("raw_results", [])
    if not raw_results:
        return state

    combined_text = "\n\n".join(d["text"] for d in raw_results)
    if state.get("running_summary"):
        state["running_summary"] += "\n\n" + combined_text
    else:
        state["running_summary"] = combined_text

    state["raw_results"] = []
    state.setdefault("messages", []).append(
        HumanMessage(content=f"Суммаризация обновлена: {len(raw_results)} уникальных статей", role="tool")
    )
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    return state


@traced
async def finalize_summary(state: dict, runtime: Runtime[Context]):
    summary_text = state.get("running_summary", "")
    unique_sources = list({d["url"] for d in state.get("raw_results", [])} | set(state.get("url_sources", [])))

    final_message = f"Итоговая суммаризация:\n{summary_text}\n\nИсточники:\n" + "\n".join(unique_sources)
    state.setdefault("messages", []).append(HumanMessage(content=final_message, role="tool"))

    user_query = state.get("query", "")
    model_context_prompt = f"{final_message}\n\nИсходный запрос пользователя:\n{user_query}"

    client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1"
    )

    try:
        response = client.chat.completions.create(
            model=f"accounts/fireworks/models/{runtime.context.model.split('/')[-1]}",
            messages=[
                {"role": "system", "content": "Используй суммаризацию и исходный запрос для продолжения диалога, "
                                              "обязательно включи источники."},
                {"role": "user", "content": model_context_prompt}
            ],
            max_tokens=10000,
            temperature=0.5
        )
        enhanced_summary = response.choices[0].message.content
        state["messages"].append(AIMessage(content=enhanced_summary))
    except Exception:
        logger.exception("Ошибка при прогоне финальной суммаризации через модель")
        state["messages"].append(AIMessage(content=final_message))

    state["raw_results"] = []
    state["url_sources"] = []
    state["running_summary"] = None
    return state

def route_message(state: dict):
    intent = state.get("intent")
    if intent == "search":
        return "arxiv_research"
    elif intent == "analyze":
        return "analyze_node"
    else:
        return END

def route_research(state: dict, max_loops: int = 2):
    if state.get("research_loop_count", 0) < max_loops:
        return "arxiv_research"
    else:
        return "finalize_summary"

async def run_agent(user_input: str, state: dict, session_id: str):
    if isinstance(user_input, str) and "exit_analysis" in user_input:
        state["analysis_mode"] = False
        state["current_article"] = None
        state["intent"] = "qa"
        logger.info(f"🔄 exit_analysis: анализ сброшен для session_id={session_id}")
        return state

    ttl_config = TTLConfig(ttl_seconds=1000)
    async with AsyncPostgresStore.from_conn_string(DB_URI, ttl=ttl_config) as store, \
               AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:

        await store.setup()
        await checkpointer.setup()
        ttl_task = await store.start_ttl_sweeper(sweep_interval_minutes=5)

        try:
            context = Context(user_id=session_id)
            graph_builder = StateGraph(State, context_schema=Context)

            # --- Узлы ---
            graph_builder.add_node(call_model)
            graph_builder.add_node(store_memory)
            graph_builder.add_node(analyze_node)
            graph_builder.add_node(arxiv_research)
            graph_builder.add_node(summarize_sources)
            graph_builder.add_node(finalize_summary)

            # --- Ребра ---
            graph_builder.add_edge("__start__", "call_model")
            graph_builder.add_conditional_edges(
                "call_model",
                lambda s: "store_memory" if s.get("intent") != "qa" else END,
                ["store_memory", END]
            )
            graph_builder.add_conditional_edges(
                "store_memory",
                route_message,
                ["arxiv_research", "analyze_node", END]
            )
            graph_builder.add_edge("arxiv_research", "summarize_sources")
            graph_builder.add_conditional_edges(
                "summarize_sources",
                lambda s: route_research(s),
                ["arxiv_research", "finalize_summary"]
            )
            graph_builder.add_edge("analyze_node", "store_memory") 
            graph_builder.add_edge("finalize_summary", END)
            graph_builder.add_edge("store_memory", END)

            graph = graph_builder.compile(
                store=store,
                checkpointer=checkpointer,
                cache=InMemoryCache()
            )
            graph.name = "MVPAgent"

            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "query": user_input,
                "current_article": state.get("current_article"),
                "analysis_mode": state.get("analysis_mode", False),
                "intent": "analyze" if state.get("analysis_mode") else state.get("intent", "qa"),
            }

            result = await graph.ainvoke(
                initial_state,
                context=context,
                config={
                    "thread_id": session_id,
                    "checkpoint_ns": "user_session",
                    "store": store,
                }
            )

            return result

        finally:
            stopped = await store.stop_ttl_sweeper(timeout=5)
            logger.info(f"TTL sweeper stopped: {stopped}")