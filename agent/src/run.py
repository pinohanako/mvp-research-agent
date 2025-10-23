from .state import State
from .context import Context
from .prompts import Intent, Filter, ArticleInfo
from .tracing import traced
from .utils import logger
from agent.src import tools
from agent.src.rag.utils import get_embeddings

from openai import OpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.cache.memory import InMemoryCache
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore, TTLConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from qdrant_client import QdrantClient, models

import os
import asyncio
from datetime import datetime
from typing import cast
from collections import defaultdict
import aiohttp
import feedparser
from dotenv import load_dotenv

DB_URI = f"postgresql://{os.environ['POSTGRES_USER']}:" \
         f"{os.environ['POSTGRES_PASSWORD']}@" \
         f"{os.environ['POSTGRES_HOST']}:" \
         f"{os.environ['POSTGRES_PORT']}/" \
         f"{os.environ['POSTGRES_DB']}?sslmode=disable&connect_timeout=10"

load_dotenv()
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"

qdrant = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY
)

@traced
async def call_model(state: dict, runtime) -> dict:
    logger.info(f"🦔 ENTER NODE: call_model")
    user_id = runtime.context.user_id
    model_str = runtime.context.model
    system_prompt_template = runtime.context.system_prompt
    state.setdefault("intent", "qa")

    model_name = f"accounts/fireworks/models/{model_str.split('/')[-1]}"

    # последние 3 сообщения для контекста памяти
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

    last_user_message = state["messages"][-1].content if state["messages"] else ""

    # intent parser + prompt
    parser = PydanticOutputParser(pydantic_object=Intent)
    intent_prompt_template = PromptTemplate(
        template=(
            "Классифицируй намерение пользователя на одну из категорий: qa, search, analyze.\n"
            "если запрос явно относится к поиску статей — \"search\",\n"
            "если это запрос на анализ или суммаризацию — \"analyze\".\n"
            "если это вопрос, на который нужен прямой ответ — \"qa\",\n"
            "Если запрос не подходит ни под одну категорию (например, приветствие, благодарность, простой диалог) — \n"
            "тоже возвращай \"qa\" как общий intent для свободного текста.\n\n"
            "Верни ТОЛЬКО один JSON-объект, соответствующий схеме ниже, без комментариев.\n"
            "Категории intent: search, qa, analyze\n\n"
            "Схема для JSON-ответа:\n"
            "{format_instructions}\n\n"
            "Запрос пользователя:\n"
            "{query}"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    raw_content = None
    intent_str = "qa"
    intent_attempts = 3
    for attempt in range(intent_attempts):
        try:
            intent_response = client.chat.completions.create(
                model=model_name,
                messages=messages_payload + [{"role": "user", "content": intent_prompt_template.format(query=last_user_message)}],
                max_tokens=100,
                temperature=0.0,
            )

            try:
                logger.debug(f"Intent response dump (attempt {attempt}): {repr(intent_response)}")
            except Exception:
                logger.debug("Intent response received (cannot repr)")

            choice = None
            if getattr(intent_response, "choices", None):
                choice = intent_response.choices[0]
            if choice is not None:
                raw_content = None
                if hasattr(choice, "message") and getattr(choice.message, "content", None):
                    raw_content = choice.message.content
                elif getattr(choice, "text", None):
                    raw_content = choice.text
                else:
                    raw_content = str(choice)
            else:
                raw_content = None

            if not raw_content:
                raise ValueError("Empty intent_response content")

            try:
                intent_parsed = parser.parse(raw_content)
                intent_str = intent_parsed.intent or "qa"
                break
            except Exception:
                logger.warning("Intent parser failed; raw_content=%s", raw_content)
                raw_content = None
        except Exception as e:
            logger.exception("Intent detection API error (attempt %d): %s", attempt, e)
            raw_content = None

    if not raw_content:
        logger.error("Intent detection failed after retries; falling back to 'qa'")
        intent_str = "qa"
        reply_content = "Произошла ошибка при обработке API запроса."
        state["intent"] = intent_str
        state["messages"].append(AIMessage(content=reply_content))
        return state

    state["intent"] = intent_str

    reply_content = None

    if intent_str == "search":
        state["query"] = last_user_message
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
            reply_content = None
        except Exception:
            logger.exception("Ошибка при генерации фильтров для поиска")
            state["title"] = None
            state["authors"] = []
            state["doi"] = None
            state["current_article"] = None
            reply_content = "Ошибка при генерации фильтров для поиска"

    elif intent_str == "analyze":
        current_article = state.get("current_article")

        if not current_article:
            state["intent"] = "qa"
            reply_content = (
                "Чтобы провести анализ, сначала нужно загрузить или выбрать статью."
            )
        else:
            for key in ("pdf_id", "pdf_url", "pdf_name"):
                if key in current_article and current_article[key] is not None:
                    state[key] = current_article[key]

            state["intent"] = "analyze"
            state["query"] = last_user_message

            logger.info(
                f"intent=analyze: переданы базовые ключи"
                f"{[k for k in ('pdf_id', 'pdf_url', 'pdf_name') if current_article.get(k)]}"
            )
            reply_content = None

    else:
        try:
            qa_response = client.chat.completions.create(
                model=model_name,
                messages=messages_payload,
                max_tokens=4999,
                temperature=0.5,
            )
            reply_content = qa_response.choices[0].message.content
        except Exception:
            logger.exception("QA call failed")
            reply_content = "Произошла ошибка при обработке API запроса."

    try:
        await tools.upsert_memory(
            content=last_user_message,
            context=f"Ответ создан {datetime.now().isoformat()}",
            user_id=user_id,
            store=runtime.store
        )
    except Exception:
        logger.exception("upsert_memory failed")

    if reply_content:
        state["messages"].append(AIMessage(content=reply_content))
    logger.info(f"🧩 Updated state.intent={state['intent']}")
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
            {"role": "system", "content": "Используй только предоставленный текст для извлечения метаданных."},
            {"role": "user", "content": prompt_template.format(first_page_text=first_page_text)}
        ],
        max_tokens=1000,
        temperature=0.0,
    )

    raw_json = response.choices[0].message.content
    article_info = parser.parse(raw_json)
    return article_info

@traced
async def analyze_node(state: dict, runtime: Runtime[Context], top_k: int = 5):
    """
    Анализирует текущую статью:
    - получает embedding запроса через Jina
    - ищет топ-N релевантных чанков в Qdrant
    - конструирует единый промпт и вызывает LLM
    """
    current_article = state.get("current_article")
    if not current_article:
        logger.warning("analyze_node: current_article отсутствует, fallback в qa")
        state["intent"] = "qa"
        state["messages"].append(
            AIMessage(content="Чтобы провести анализ, сначала нужно загрузить или выбрать статью.")
        )
        return state

    # 1. embedding запроса
    query_vector = get_embeddings([state.get("query", "")], task="retrieval.query")[0]

    # 2. Поиск топ-N релевантных чанков в Qdrant
    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="pdf_id",
                match=models.MatchValue(value=current_article["pdf_id"])
            )]
        ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=top_k
    )

    chunks = [p.payload["document"] for p in results.result]

    # 3. Формируем промпт для LLM
    article_summary = (
        f"Название статьи: {current_article.get('title')}\n"
        f"Аннотация: {current_article.get('summary')}\n"
        f"Авторы статьи: {current_article.get('authors')}\n"
        f"Дата публикации статьи: {current_article.get('published')}\n"
    )
    chunks_text = "\n\n".join(chunks)
    query_text = state.get("query", "")
    prompt_text = (
        f"Учитывай контекст при ответе на запрос пользователя:\n\n"
        f"Извлеченные фрагменты статьи:\n{chunks_text}\n\n"
        f"Общий контекст о статье:\n{article_summary}\n\n"
        f"Запрос пользователя: {query_text}"
    )

    client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1"
    )
    try:
        response = client.chat.completions.create(
            model=f"accounts/fireworks/models/{runtime.context.model.split('/')[-1]}",
            messages=[
                {"role": "system", "content": "Ты научный ассистент. Используй предоставленные данные о статье и контекст запроса."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=1500,
            temperature=0.5
        )
        answer = response.choices[0].message.content
        state["messages"].append(AIMessage(content=answer))
        logger.info("analyze_node: ответ от LLM добавлен в state")
    except Exception:
        logger.exception("analyze_node: ошибка вызова LLM")
        state["messages"].append(AIMessage(content="Ошибка при генерации ответа."))

    return state

MAX_RESULTS_PER_QUERY = 5
@traced
async def arxiv_research(state: dict, runtime):
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
async def summarize_sources(state: dict, runtime):
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
async def finalize_summary(state: dict, runtime):
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
                {"role": "system", "content": "Используй суммаризацию и исходный запрос для продолжения диалога, обязательно включи источники."},
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
    ttl_config = TTLConfig(ttl_seconds=1000)
    
    async with AsyncPostgresStore.from_conn_string(DB_URI, ttl=ttl_config) as store, \
               AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:

        await store.setup()
        await checkpointer.setup()
        ttl_task = await store.start_ttl_sweeper(sweep_interval_minutes=5)

        try:
            context = Context(user_id=session_id)
            graph_builder = StateGraph(State, context_schema=Context)

            # --- Узлы
            graph_builder.add_node(call_model)
            graph_builder.add_node(store_memory)
            graph_builder.add_node(analyze_node)
            graph_builder.add_node(arxiv_research)
            graph_builder.add_node(summarize_sources)
            graph_builder.add_node(finalize_summary)

            # --- Ребра
            graph_builder.add_edge("__start__", "call_model")
            graph_builder.add_conditional_edges(
                "call_model",
                lambda state: "store_memory" if state.get("intent") != "qa" else END,
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
                lambda state: route_research(state),
                ["arxiv_research", "finalize_summary"]
            )
            graph_builder.add_edge("finalize_summary", END)
            graph_builder.add_edge("analyze_node", "call_model")

            graph = graph_builder.compile(store=store, checkpointer=checkpointer, cache=InMemoryCache())
            graph.name = "MVPAgent"

            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "query": user_input,
                "current_article": state.get("current_article"), 
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
            print(f"TTL sweeper stopped: {stopped}")
