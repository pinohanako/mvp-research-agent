from __future__ import annotations
from operator import add

from typing import Optional, List
from dataclasses import dataclass, field

from typing_extensions import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage

# Cтатьи, загруженные пользователем
class ArticleState(TypedDict):
    title: Optional[str]
    summary: Optional[str]        # аннотация
    authors: Optional[List[str]]
    published: Optional[str]
    pdf_url: Optional[str]
    pdf_id: Optional[str]
    pdf_name: Optional[str]
    doi: Optional[str]

# Контекст пользователя — текущий запрос, интент, статья
class PrivateState(TypedDict):
    query: Optional[str]
    intent: Optional[str]
    current_article: Optional[ArticleState]

# Временная исследовательская память — результаты поиска
class SummaryState(TypedDict):
    # Тексты из найденных статей
    raw_results: List[str] 
    
    # URL источников
    url_sources: List[str]
    
    # Количество итераций сбора данных
    research_loop_count: int
    
    # Текущая агрегированная суммаризация
    running_summary: Optional[str]

class OverallState(PrivateState, ArticleState, SummaryState):
    pass

@dataclass(kw_only=True)
class State(OverallState):
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