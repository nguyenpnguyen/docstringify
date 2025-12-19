import weave
import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    answer_similarity
)

# --- 1. Setup Vertex AI (Gemini) as the Judge ---
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# Configure your Google Cloud Project
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"
PROJECT_ID = "your-gcp-project-id"  # <--- REPLACE THIS
REGION = "us-central1"


def get_google_metrics():
    """Configures RAGAS metrics to use Gemini Pro via Vertex AI."""

    # Initialize Vertex AI Models
    vertex_llm = ChatVertexAI(
        model_name="gemini-pro",
        project=PROJECT_ID,
        location=REGION
    )
    vertex_embeddings = VertexAIEmbeddings(
        project=PROJECT_ID,
        location=REGION
    )

    # Select relevant metrics
    metrics = [
        faithfulness,  # Hallucination check
        answer_correctness,  # Semantic + Factual match against Ground Truth
        answer_similarity  # Pure Embedding similarity
    ]

    # Inject Gemini into Ragas metrics
    for m in metrics:
        if hasattr(m, 'llm'):
            m.llm = vertex_llm
        if hasattr(m, 'embeddings'):
            m.embeddings = vertex_embeddings

    return metrics


# --- 2. Load Local Dataset ---
def load_fixed_dataset(file_path="test_dataset.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ {file_path} not found. Please run 'prepare_data.py' first.")

    print(f"📂 Loading dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- 4. Main Execution ---
def main():
    pass


if __name__ == "__main__":
    main()
