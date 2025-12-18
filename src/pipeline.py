import weave
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import your loader logic (assuming you have src/loader.py from previous steps)
from src.loader import load_llm, load_embeddings
from src.retrievers import retrieve_code_snippets

# Define the Prompt
PROMPT = PromptTemplate.from_template("""
You are an expert Python documentation generator.
Below is a python function that needs a docstring.
I have also provided some CONTEXT (other related functions or usages) to help you understand it.

--- CONTEXT ---
{context}

--- TARGET FUNCTION ---
{function_code}

--- INSTRUCTIONS ---
Generate a Google-style docstring for the target function. 
Return ONLY the docstring (wrapped in triple quotes). Do not return the code.
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@weave.op()
def docstring_rag_pipeline(target_code: str):
    """
    The main RAG pipeline: Retrieve -> Generate
    """
    # 1. Load Resources
    llm = load_llm()
    embeddings = load_embeddings()
    retriever = retrieve_code_snippets(embeddings, k_semantic=4)

    # 2. Retrieval Step
    retrieved_docs = retriever.invoke(target_code)

    # 3. Generation Step
    chain = (
            {"context": lambda x: format_docs(retrieved_docs), "function_code": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
    )

    result = chain.invoke(target_code)

    return {
        "result": result,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs]
    }
