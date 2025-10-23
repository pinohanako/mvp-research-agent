import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated
from agent.src.memory_agent import prompts

@dataclass(kw_only=True)
class Context:
    user_id: str = "default"
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="fireworks/gpt-oss-120b",
        metadata={
            "description": "Название LLM, которую будет использовать агент "
            "Должно быть вида: provider/model-name."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self):
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
