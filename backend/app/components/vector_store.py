# vector_store.py
import os
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

# Ensure DB_FAISS_PATH is absolute
DB_FAISS_PATH = os.path.abspath(DB_FAISS_PATH)

def load_vector_store():
    """
    Load existing FAISS vectorstore if present.
    Returns: FAISS vectorstore object or None
    """
    try:
        embedding_model = get_embedding_model()

        if os.path.exists(DB_FAISS_PATH) and os.path.isdir(DB_FAISS_PATH):
            logger.info("Loading existing vectorstore from %s", DB_FAISS_PATH)
            db = FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Vector store loaded successfully")
            return db
        else:
            logger.warning("Vectorstore not found at %s", DB_FAISS_PATH)
            return None

    except Exception as e:
        error_message = CustomException("Failed to load vectorstore", e)
        logger.error(str(error_message))
        return None

def save_vector_store(text_chunks):
    """
    Create and save a FAISS vectorstore from text chunks.
    Returns: FAISS vectorstore object or None
    """
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to create vectorstore")

        logger.info("Initializing embedding model for vectorstore...")
        embedding_model = get_embedding_model()

        logger.info("Creating new FAISS vectorstore from %d chunks", len(text_chunks))
        db = FAISS.from_documents(text_chunks, embedding_model)

        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        db.save_local(DB_FAISS_PATH)
        logger.info("✅ Vectorstore saved successfully at %s", DB_FAISS_PATH)

        return db

    except Exception as e:
        error_message = CustomException("Failed to create/save vectorstore", e)
        logger.error(str(error_message))
        return None

def ensure_vector_store(text_chunks=None):
    """
    Load existing vectorstore or create a new one if missing.
    Optional: provide text_chunks to create a new vectorstore.
    Returns: FAISS vectorstore object or None
    """
    db = load_vector_store()
    if db is None:
        logger.info("Vectorstore missing, attempting to create a new one...")
        if text_chunks:
            db = save_vector_store(text_chunks)
            if db:
                logger.info("✅ Vectorstore created successfully")
            else:
                logger.error("❌ Failed to create vectorstore")
        else:
            logger.warning("No text chunks provided; cannot create vectorstore")
    return db

db = load_vector_store()
if db:
    print(f"VectorStore type: {type(db)}, Number of docs: {len(db.index_to_docstore_id)}")
