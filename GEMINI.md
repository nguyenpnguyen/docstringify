# Gemini Context: `docstringify`

This document provides a comprehensive overview of the `docstringify` project, its architecture, and operational procedures to serve as a guide for AI-driven development.

## Project Overview

`docstringify` is a command-line tool that automatically generates high-quality Python docstrings by analyzing the codebase's structure. It uses a database-centric approach to understand code relationships.

The tool works in two main phases:
1.  **Indexing:** It first traverses a project directory, parsing all Python files using the `ast` module to identify functions and classes. These code objects and their relationships (i.e., which functions call which other functions) are stored in a SQLite database using Peewee. This creates a structured, queryable model of the codebase, including a full call graph.
2.  **Generation:** When generating a docstring for a target piece of code, the tool retrieves relevant context by querying the database for its direct dependencies (functions it calls) and dependents (functions that call it). This structural context, along with the target code, is then fed to a Large Language Model (via LangChain and Ollama) to generate an accurate and context-aware Google-style docstring.

### Key Technologies

*   **Core Logic:** Python (>=3.12)
*   **Database:** Peewee (for SQLite)
*   **AI/ML Frameworks:** `langchain`, `langchain-ollama`
*   **CLI:** `typer`
*   **Evaluation & Tracing:** `weave`, `ragas`, using Gemini from Vertex AI as the judge.
*   **Packaging:** `setuptools`
*   **Dependency Management:** `uv`

## Building and Running

### Installation

The project uses `uv` for dependency management.

1.  **Install Dependencies:**
    ```bash
    uv pip install -e .
    ```
2.  **Install Development Dependencies (for evaluation):**
    ```bash
    uv pip install -e ".[dev]"
    ```

### Running the Tool

The main functionality is exposed through the `docstringify` command.

```bash
# Generate docstrings for a project
docstringify /path/to/your/python/project
```

### Running Evaluation

The project uses `ragas` and `weave` to evaluate the quality of the generated docstrings against a test dataset (`test_dataset.json`). The evaluation script uses Gemini on Vertex AI as a "judge" to score the outputs.

1.  **Set up Authentication:** Ensure you are authenticated with Google Cloud:
    ```bash
    gcloud auth application-default login
    ```

2.  **Run the script:**
    ```bash
    python evaluate.py
    ```

The results, including scores for faithfulness and correctness, will be printed to the console and published to a Weave dashboard.

## Development Conventions

*   **Project Structure:** The code follows a standard `src` layout.
*   **Linting/Formatting:** The presence of a `.ruff_cache` directory suggests `ruff` is used for code linting and formatting.
*   **Testing:** `pytest` is listed as a development dependency, indicating it is the testing framework.
*   **Notebooks:** The project includes Jupyter notebooks (`dataset.ipynb`) for data exploration and preparation.