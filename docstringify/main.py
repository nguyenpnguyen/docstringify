import logging
from pathlib import Path

import typer
from docstringify.agent import agent, AgentState
from docstringify.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def docstringify(project_path: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),):
    """
    Automatically generates docstrings for a Python project.
    """
    project_path = project_path.resolve()
    db_path = project_path / "autodoc.db"
    
    # Ensure database is initialized with the correct path
    init_db(str(db_path))
    
    initial_state = AgentState(
        db_path=str(db_path),
        queue=[],
        current_job_id=None,
        current_code=None,
        retrieved_context="",
        generated_docstring=None,
        file_docstring_changes={},
    )
    
    logger.info(f"Starting the docstring generation process for {project_path}...")
    agent.invoke(initial_state, config={"recursion_limit": 50})
    logger.info("Docstring generation process finished.")


if __name__ == "__main__":
    app()
