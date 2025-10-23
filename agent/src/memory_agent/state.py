from __future__ import annotations
from operator import add

from dataclasses import dataclass, field

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated
from typing_extensions import TypedDict
from typing import Optional, List, Dict

# Долговременная память — подтвержденные статьи, загруженные пользователем
class ArticleState(TypedDict):
    title: Optional[str]
    summary: Optional[str]        # аннотация
    authors: Optional[List[str]]
    published: Optional[str]
    pdf_url: Optional[str]
    pdf_id: Optional[str]
    pdf_name: Optional[str]
    doi: Optional[str]

# Контекст пользователя — текущий запрос, интент, выбранная статья
class PrivateState(TypedDict):
    query: Optional[str]
    intent: Optional[str]
    current_article: Optional[ArticleState]

# Временная исследовательская память — результаты поиска
class SummaryState(TypedDict):
    raw_results: List[str]
    """Тексты из найденных статей"""

    url_sources: List[str]
    """Ссылки на источники (URL)"""

    research_loop_count: int
    """Количество итераций сбора данных"""

    running_summary: Optional[str]
    """Текущая агрегированная суммаризация"""

class OverallState(PrivateState, ArticleState, SummaryState):
    pass

@dataclass(kw_only=True)
class State(OverallState):
    # История диалога
    messages: Annotated[List[AnyMessage], add_messages] = field(default_factory=list)

    # Контекст текущего взаимодействия
    query: Optional[str] = None
    intent: Optional[str] = None                                   
    current_article: Optional[ArticleState] = None

    # Временное исследовательское состояние
    raw_results: Annotated[List[str], add] = field(default_factory=list)
    url_sources: Annotated[List[str], add] = field(default_factory=list)
    research_loop_count: int = 0
    running_summary: Optional[str] = None

__all__ = [
    "State",
]