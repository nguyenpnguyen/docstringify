from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    llm_id: str = "qwen3:4b-instruct"
    temperature: float = 0.2
    num_ctx: int = 8192
    db_name: str = "docstringify.db"

# Global config instance that can be updated
settings = Config()

def update_settings(llm_id: Optional[str] = None, temperature: Optional[float] = None):
    if llm_id:
        settings.llm_id = llm_id
    if temperature is not None:
        settings.temperature = temperature
