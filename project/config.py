# --- Directory Configuration ---
MARKDOWN_DIR = "markdown_docs"
PARENT_STORE_PATH = "parent_store"
QDRANT_DB_PATH = "qdrant_db"

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"

# --- Model Configuration ---
DENSE_MODEL = "text-embedding-v4"
LLM_MODEL = "qwen3-max"
LLM_TEMPERATURE = 0

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 50
MIN_PARENT_SIZE = 1000
MAX_PARENT_SIZE = 5000
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- LangGraph Configuration ---
MAX_LOCAL_RETRIES = 2
