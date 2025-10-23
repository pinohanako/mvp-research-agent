import uuid
from uuid import uuid4
from typing import Annotated

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore

async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """ Добавляет в DB
    Args:
        content: Основное содержимое воспоминания. Например:
            "Пользователь выразил заинтересованность в изучении монархического строя.""
        context: Дополнительный контекст для воспоминания. Например:
            "Это было упомянуто при обсуждении государственногой устройства.""
        memory_id: Для обновления памяти
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"