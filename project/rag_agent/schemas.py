from typing import List
from pydantic import BaseModel, ConfigDict, Field

class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra='allow')

    is_clear: bool = Field(
        description="用户问题是否清晰可回答。清晰则为true，需要澄清则为false。"
    )
    questions: List[str] = Field(
        default_factory=list,
        description="改写后的独立问题列表。仅当is_clear为true时填写，否则留空列表。"
    )
    clarification_needed: str = Field(
        default="",
        description="仅当is_clear为false时填写，说明需要用户补充哪些信息。若is_clear为true，必须为空字符串。"
    )
