# vector_database.py
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

PDF_DIR = "pdfs"
VECTORSTORE_DIR = os.path.join("vectorstore", "db")

# Embedding model (separate from LLM)
# Use an embedding model supported by Ollama, e.g. "nomic-embed-text"
def get_embedder_model_name():
    return "nomic-embed-text"

def load_pdf(path):
    """
    Load PDF and return list of LangChain Document objects.
    path: local filesystem path to a PDF
    """
    loader = PDFPlumberLoader(path)
    return loader.load()

def split_docs(documents, chunk_size=800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(documents)

def create_faiss_db_from_docs(documents, persist_path=VECTORSTORE_DIR):
    """
    Create FAISS vectorstore from LangChain documents and save locally.
    """
    os.makedirs(persist_path, exist_ok=True)

    emb_model_name = get_embedder_model_name()
    embeddings = OllamaEmbeddings(model=emb_model_name)

    db = FAISS.from_documents(documents, embeddings)
    # save_local writes the index and metadata into persist_path
    db.save_local(persist_path)
    return db

def load_faiss_db(persist_path=VECTORSTORE_DIR):
    """
    Loads an existing FAISS DB. Raises FileNotFoundError if not present.
    """
    if not os.path.isdir(persist_path):
        raise FileNotFoundError(f"FAISS folder not found: {persist_path}")

    emb_model_name = get_embedder_model_name()
    embeddings = OllamaEmbeddings(model=emb_model_name)

    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
