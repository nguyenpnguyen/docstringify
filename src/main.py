import logging
from pathlib import Path

import typer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def docstringify(project_path: Path = typer.Argument(..., exists=True, file_okay=False),):
    pass

@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}!")

if __name__ == "__main__":
    app()
