import logging
from pathlib import Path
from typing import Optional

import typer
from docstringify.workflow import workflow 
from docstringify.nodes import ApplicationState
from docstringify.db import init_db
from docstringify.config import settings, update_settings

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def docstringify(
    project_path: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    llm_id: Optional[str] = typer.Option(None, "--llm-id", "--llm", help="The Ollama model ID to use."),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="The temperature for LLM generation."),
):
    """
    Automatically generates docstrings for a Python project.
    """
    # Update global settings with CLI arguments
    update_settings(llm_id=llm_id, temperature=temperature)
    
    project_path = project_path.resolve()
    db_path = project_path / settings.db_name
    
    logger.info(f"Using LLM: {settings.llm_id} (temp: {settings.temperature})")
    
    # Ensure database is initialized with the correct path
    init_db(str(db_path))
    
    initial_state = ApplicationState(
        db_path=str(db_path),
        queue=[],
        current_job_id=None,
        current_code=None,
        retrieved_context="",
        generated_docstring=None,
        file_docstring_changes={},
    )
    
    logger.info(f"Starting the docstring generation process for {project_path}...")
    workflow.invoke(initial_state, config={"recursion_limit": 100})
    logger.info("Docstring generation process finished.")


if __name__ == "__main__":
    app()
