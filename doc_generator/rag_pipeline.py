from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models import LanguageModelLike
from langchain_core.vectorstores import VectorStoreRetriever

def create_rag_chain(llm: LanguageModelLike, retriever: VectorStoreRetriever):
    """
    Creates the full RAG chain for docstring generation.
    """
    
    # This prompt template is key to your RAG performance
    template = """
    You are an expert Python programmer. Your job is to write a clear, concise, 
    and professional Google-style docstring for the given Python code.

    Use the following context from the rest of the codebase to help you
    understand the code's purpose and how it's used.

    CONTEXT:
    {context}

    CODE TO DOCUMENT:
    {input}

    GENERATED DOCSTRING:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # This chain "stuffs" the retrieved documents into the {context}
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # This final chain combines the retriever and the document_chain
    # It takes the {input}, passes it to the retriever to get {context},
    # then passes {input} and {context} to the document_chain.
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    print("RAG chain created.")
    return rag_chain
