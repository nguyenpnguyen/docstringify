import weave
import os
import json
import logging
from typing import List, Annotated

import typer

from pydantic import BaseModel

# Ragas & LangChain
from langchain_core.documents import Document
from datasets import Dataset
from ragas import experiment
from ragas.metrics.collections import AnswerRelevancy

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from google import genai

from agent import agent
from retrievers import index_codebase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

llm = llm_factory(
    "gemini-flash-latest",
    provider="google",
    client=client,
    temperature=0,
)

embeddings = embedding_factory("google", model="gemini-embedding-001", client=client)

def load_fixed_dataset(file_path="test_dataset.json", num_samples: int = None):
    """
Load a fixed evaluation dataset from a JSON file.

This function reads a dataset stored in a JSON file and optionally limits the number of samples to evaluate.

Args:
    file_path (str): Path to the JSON file containing the dataset. Defaults to "test_dataset.json".
    num_samples (int): Number of samples to load from the dataset. If None, loads all samples.

Returns:
    list: A list of dictionaries, each representing a sample with keys like 'source_code' and 'docstring'.

Raises:
    FileNotFoundError: If the specified file path does not exist.

Example:
    >>> data = load_fixed_dataset("test_dataset.json", num_samples=10)
    >>> print(len(data))  # Output: 10
"""
    if not os.path.exists(file_path):
        """Index a list of documents using the agent's embeddings to create a vector store for retrieval.

Args:
    documents (List[Document]): A list of Document objects to be indexed.
        """Predicts a docstring for a given question by invoking an agent and retrieving relevant context.

Args:
    question (str): The input question or code snippet for which a docstring is to be generated.

Returns:
    dict: A dictionary containing:
        - 'result' (str): The generated docstring based on the input question.
        - 'retrieved_contexts' (list[str]): A list of strings representing the retrieved context documents (page content) used to generate the docstring.
"""

Returns:
    """Class representing the result of an experiment, containing the relevancy score of an answer to a user question.

Attributes:
    answer_relevancy (float): The score indicating how relevant the answer is to the user's question, ranging from 0 to 1.
        """Run evaluation of answer relevancy using a specified LLM and embeddings.

This function scores the relevance of a model's generated answer to the provided question
using the AnswerRelevancy metric. It takes a row of evaluation data containing the
question and answer, and returns an ExperimentResult object with the computed
relevancy score.

Args:
    row: A dictionary containing 'question' and 'answer' keys, each with a list of
         strings. The first element of each list is used for evaluation.

Returns:
    ExperimentResult: An object containing the relevancy score as a float value.
        """Main function to evaluate a dataset using a LangGraph agent and compute RAGAS scores.

This function loads a fixed evaluation dataset (from a JSON file), evaluates each sample by running the DocstringEvaluatorModel to generate predictions, and then assesses the relevance of the generated answers against the ground truth docstrings. It computes RAGAS scores using a Gemini Judge model and publishes the final evaluation results to the Weave platform for visualization.

Args:
    num_samples (int, optional): Number of samples to evaluate from the dataset. If None, evaluates all samples. Defaults to None.

Returns:
    None: The function logs the evaluation progress and final results, and publishes them to Weave. No explicit return value.

Example:
    python evaluate.py --samples 100
    # Evaluates the first 100 samples from the dataset and publishes results to Weave.
"""
"""
"""
    The result of indexing the documents using the agent's embeddings, typically a vector store or index object.
"""
        raise FileNotFoundError(f"❌ {file_path} not found. Please run 'prepare_data.py' first.")
        logger.error(f"{file_path} not found. Please run 'prepare_data.py' first.")

    logger.info(f"Loading evaluation dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]

    logger.info(f"Loaded {len(data)} samples for evaluation.")
    return data


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

class ExperimentResult(BaseModel):
    answer_relevancy: float

@experiment(ExperimentResult)
def run_eval(row) -> ExperimentResult:
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    relevancy_result = answer_relevancy.score(
        user_input=row["question"][0],
        response=row["answer"][0]
    )

    return ExperimentResult(answer_relevancy=relevancy_result.value)

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

    try:
        eval_data = load_fixed_dataset(num_samples=num_samples)
    except FileNotFoundError as e:
        print(e)
        return

    logger.info(f"Running inference with {len(eval_data)} samples.")

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

    logger.info("Calculating RAGAS Scores with Gemini Judge...")

    dataset = Dataset.from_list(results)

    exp_results = run_eval(dataset)

    logger.info("Experiment Results:")
    logger.info(exp_results)

    # results_df = exp_results.to_pandas()

    logger.info("Publishing results to Weave...")
    # weave.publish(results_df, "ragas_evaluation_table")

    logger.info("✅ Done! Check your Weave dashboard.")


if __name__ == "__main__":
    app()
