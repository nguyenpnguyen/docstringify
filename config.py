# --- Model IDs ---
# The base model for the generator
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"

# The adapter path for your finetuned model (set this after finetuning)
# ADAPTER_PATH = "./qwen3_finetuned_adapters" 
ADAPTER_PATH = None # Keep as None for baseline run

# The model used to create vector embeddings
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# --- Paths ---
# The path to the codebase you want to document
CODEBASE_PATH = "./codebase_to_doc"

# The path to persist the ChromaDB vector store
CHROMA_STORE_PATH = "./chroma_db"

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "Doc_Generator"
MLFLOW_TRACKING_URI = "file:./mlruns" # Log to a local ./mlruns folder
