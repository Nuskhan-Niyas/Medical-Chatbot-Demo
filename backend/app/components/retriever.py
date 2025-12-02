from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from app.components.llm import load_llm, GemmaLLM
from app.components.vector_store import load_vector_store, save_vector_store
from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import torch


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """Answer the following medical question in 6-10 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def create_qa_chain():
    """Load or create VectorStore + LLM + QA Chain"""
    logger.info("🔄 Creating QA Chain...")
    try:
        # Load or create vectorstore
        db = load_vector_store()
        if db is None:
            logger.warning("Vector store not found. Creating a new one...")
            documents = load_pdf_files()
            if not documents:
                raise CustomException("No documents found to create vector store")
            text_chunks = create_text_chunks(documents)
            db = save_vector_store(text_chunks)
            if db is None:
                raise CustomException("Failed to create vector store")

        logger.info(f"✅ Vector store loaded/created successfully.")

        # Load Gemma LLM
        tokenizer, llm_model = load_llm()
        if not tokenizer or not llm_model:
            raise CustomException("LLM not loaded")

        llm = GemmaLLM(tokenizer=tokenizer, model=llm_model)

        # Create retriever
        retriever: BaseRetriever = db.as_retriever(search_kwargs={"k": 1})

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": set_custom_prompt()},
            input_key="query"
        )

        logger.info("✅ QA chain created successfully")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to create QA chain", e)
        logger.error(str(error_message))
        return None


# Standalone test
if __name__ == "__main__":
    qa_chain = create_qa_chain()
    if qa_chain:
        print("✅ QA chain created successfully")
        test = qa_chain({"query": "What are the symptoms of diabetes?"})
        print("Test output:", test.get("result", "No result returned"))
    else:
        print("❌ QA chain could not be created")


from langchain.callbacks.manager import CallbackManagerForLLMRun

def _call(self, prompt: str, stop=None, run_manager: CallbackManagerForLLMRun = None) -> str:
    """Single-prompt generation with streaming callback support"""
    inputs = self.tokenizer(prompt, return_tensors="pt")

    output_text = ""
    with torch.no_grad():
        for token_id in self.model.generate(**inputs, max_new_tokens=self.max_new_tokens):
            decoded = self.tokenizer.decode(token_id, skip_special_tokens=True)
            if run_manager:
                run_manager.on_llm_new_token(decoded)
            output_text += decoded
    return output_text
