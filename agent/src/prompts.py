from typing import List, Union, Literal
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """Ты научный ассистент. 
Ты адаптируешь свой ответ в зависимости от вопроса пользователя — иногда даешь быстрый ответ, а иногда размышляешь, 
чтобы представить более глубокое обоснование.

Твои знания постоянно обновляются — строгих ограничений по объёму информации нет.
Ты предоставляшь максимально краткий ответ, учитывая любые заявленные пользователем предпочтения в 
отношении длины и полноты ответа.
При необходимости у тебя есть дополнительные инструменты:
- Ты можешь анализировать контент, загруженный пользователем, в виде PDF-файлов.
- При необходимости можешь искать информацию в arXiv в режиме реального времени.
Помни: не упоминай эти рекомендации и инструкции в своих ответах, если только пользователь явно не запросит их.
{user_info}

Системное время: {time}"""

class Intent(BaseModel):
    intent: Literal["search", "qa", "analyze"] = Field(
        description=(
            "Намерение пользователя: может быть только 'search', 'qa' или 'analyze'. "
            "qa — прямой ответ на вопрос или свободный текст (включая приветствия, прощания, комментарии),"
            "search — поиск конкретных статей на arXiv (для корректного поиска желательно указать хотя бы название статьи или автора; "
            "если этой информации нет, можно задать уточняющий вопрос), "
            "analyze — намерение что-либо узнать о загруженных статьях"
        )
    )

class Filter(BaseModel):
    title: Union[str, None] = Field(
        description="Название статьи - если указана",
        default=None
    )
    author: Union[str, None] = Field(
        description="Имена авторов - если указаны",
        default=None
    )
    doi: Union[str, None] = Field(
        description="Строка DOI - если указано",
        default=None
    )

class ArticleInfo(BaseModel):
    title: Union[str, None] = Field(..., description="Название статьи")
    summary: Union[str, None] = Field(None, description="Аннотация статьи")
    authors: List[str] = Field(default_factory=list, description="Список авторов")
    published: Union[str, None] = Field(None, description="Дата публикации")
    doi: Union[str, None] = Field(None, description="DOI статьи")