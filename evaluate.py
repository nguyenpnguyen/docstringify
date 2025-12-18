import weave
import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_relevancy,
    faithfulness,
    answer_correctness,
    answer_similarity
)
from src.pipeline import docstring_rag_pipeline

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
        model_name="textembedding-gecko@003",
        project=PROJECT_ID,
        location=REGION
    )

    # Select relevant metrics
    metrics = [
        context_relevancy,  # Does not need ground truth context
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


# --- 3. Weave Evaluation Model ---
class DocstringEvaluatorModel(weave.Model):
    @weave.op()
    def predict(self, question: str):
        # Run your RAG pipeline
        output = docstring_rag_pipeline(question)

        # Format for Ragas
        combined_contexts = [question] + output["retrieved_contexts"]

        return {
            "question": question,
            "answer": output["result"],
            "contexts": combined_contexts
        }


# --- 4. Main Execution ---
def main():
    weave.init("docstringify-vertex-eval")

    try:
        metrics = get_google_metrics()
    except Exception as e:
        print(f"❌ Error configuring Vertex AI: {e}")
        print("Please ensure you have set PROJECT_ID and run 'gcloud auth application-default login'")
        return

    # 1. Load Data from JSON
    try:
        eval_data = load_fixed_dataset()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"🏃 Running Inference on {len(eval_data)} samples...")
    results = []

    # Iterate through the fixed dataset
    for row in eval_data:
        # Run the pipeline
        output = docstring_rag_pipeline(row["question"])

        # Combine target code + retrieved chunks for "Context"
        full_context = [row["question"]] + output["retrieved_contexts"]

        results.append({
            "question": row["question"],
            "answer": output["result"],
            "contexts": full_context,
            "ground_truth": row["ground_truth"]
        })

    # 2. Run Evaluation
    print("📊 Calculating RAGAS Scores with Gemini Judge...")
    hf_dataset = Dataset.from_list(results)

    scores = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
    )

    print("\n--- Evaluation Results ---")
    print(scores)

    # Log to Weave
    weave.publish(scores, "vertex_ai_scores")


if __name__ == "__main__":
    main()
