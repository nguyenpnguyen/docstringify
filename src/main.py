import logging
from pathlib import Path

import typer
from src.agent import agent, AgentState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def docstringify(project_path: Path = typer.Argument(..., exists=True, file_okay=False),):
    """
    Automatically generates docstrings for a Python project.
    """
    db_path = project_path / "autodoc.db"
    
    initial_state = AgentState(
        db_path=str(db_path),
        queue=[],
        current_job_id=None,
        current_code=None,
        retrieved_context="",
        generated_docstring=None,
    )
    
    logger.info("Starting the docstring generation process...")
    agent.invoke(initial_state)
    logger.info("Docstring generation process finished.")


if __name__ == "__main__":
    app()
