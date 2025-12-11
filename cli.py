from pathlib import Path

import typer

app = typer.Typer()

@app.command()
def docstringify(
    project_path: Path = typer.Argument(..., exists=True, file_okay=False),
):
    pass

if __name__ == "__main__":
    app()
