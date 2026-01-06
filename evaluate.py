import weave
import os
import json
from typing import List, Annotated

import typer

# Ragas & LangChain
from langchain_core.documents import Document
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerCorrectness

from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings

from google import genai

from agent import agent
from retrievers import index_codebase

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def get_google_metrics():
    """
    Configures RAGAS metrics to use Gemini 2.5 Flash with the Gemini API.
    """

    llm = llm_factory(
        "gemini-2.5-flash",
        provider="google",
        client=client
    )

    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    metrics = [
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]

    return metrics


# --- 2. Load Local Dataset ---
def load_fixed_dataset(file_path="test_dataset.json", num_samples: int = None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ {file_path} not found. Please run 'prepare_data.py' first.")

    print(f"📂 Loading dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]

    print(f"Loaded {len(data)} samples.")
    return data


# --- 3. Weave Evaluation Model ---
class DocstringEvaluatorModel(weave.Model):
    """Wraps the LangGraph agent to ensure Weave traces the execution."""

    @weave.op()
    def index_code(self, documents: List[Document]):
        return index_codebase(documents, agent.embeddings)

    @weave.op()
    def predict(self, question: str):
        output_state = agent.invoke({"code_snippet": question})

        generated_docstring = output_state.get("docstring", "")
        retrieved_docs = output_state.get("context", [])
        context_strings = [doc.page_content for doc in retrieved_docs]

        return {
            "result": generated_docstring,
            "retrieved_contexts": context_strings
        }


# --- 4. Main Execution ---
app = typer.Typer()


@app.command()
def main(
        num_samples: Annotated[
            int,
            typer.Option(
                "--samples",
                "-n",
                help="Number of samples to evaluate from the dataset. Defaults to all.",
            ),
        ] = None,
):
    weave.init("docstringify-vertex-eval")

    # 1. Setup Metrics (Automatic Auth)
    try:
        metrics = get_google_metrics()
        print("✅ Vertex AI (Gemini 2.5 Flash) configured.")
    except Exception as e:
        print(f"❌ Error configuring Vertex AI: {e}")
        return

    # 2. Load Data
    try:
        eval_data = load_fixed_dataset(num_samples=num_samples)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"🏃 Running Inference on {len(eval_data)} samples...")

    # Initialize Weave Model
    model = DocstringEvaluatorModel()

    results = []

    for row in eval_data:
        question = row["source_code"]
        ground_truth = row["docstring"]

        prediction = model.predict(question)

        results.append({
            "question": question,
            "answer": prediction["result"],
            "contexts": prediction["retrieved_contexts"],
            "ground_truth": ground_truth
        })

    # 3. Run Evaluation
    print("📊 Calculating RAGAS Scores with Gemini Judge...")

    hf_dataset = Dataset.from_list(results)

    scores = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
    )

    print("\n--- Evaluation Summary ---")
    print(scores)

    results_df = scores.to_pandas()
    print("💾 Publishing detailed results to Weave...")
    weave.publish(results_df, "ragas_evaluation_table")

    print("✅ Done! Check your Weave dashboard.")


if __name__ == "__main__":
    app()
