import os
import streamlit as st

from vector_database import load_pdf, split_docs, create_faiss_db_from_docs, load_faiss_db, PDF_DIR, VECTORSTORE_DIR
from rag_pipeline import retrieve_docs_from_db, answer_query_from_docs, llm_model

st.set_page_config(page_title="AI Legal Chatbot (Ollama DeepSeek)", layout="wide")
st.title("⚖️ AI Legal Chatbot")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf", help="Upload one PDF to index")

if uploaded_file:
    pdf_path = os.path.join(PDF_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Saved: {pdf_path}")

    # Process PDF and create FAISS DB
    with st.spinner("Indexing document (creating embeddings)… this runs locally via Ollama embedding model"):
        docs = load_pdf(pdf_path)
        chunks = split_docs(docs)
        db = create_faiss_db_from_docs(chunks)
        st.success("Document indexed and vector DB created.")
        # store db object in session to avoid reloading
        st.session_state["faiss_db"] = db
else:
    st.info("Upload a PDF to start. (Or copy a PDF into the 'pdfs' folder.)")

query = st.text_area("Ask your legal question:", height=150)

if st.button("Ask AI Lawyer 🧑‍⚖️"):
    if "faiss_db" not in st.session_state:
        # try to load existing db if present
        try:
            db = load_faiss_db()
            st.session_state["faiss_db"] = db
            st.info("Loaded existing vector DB from disk.")
        except FileNotFoundError:
            st.error("Vector DB not found. Please upload and index a PDF first.")
            st.stop()
    else:
        db = st.session_state["faiss_db"]

    with st.spinner("Retrieving relevant sections…"):
        retrieved = retrieve_docs_from_db(db, query)
    with st.spinner("Generating answer (DeepSeek R1 via Ollama)…"):
        answer = answer_query_from_docs(retrieved, query)
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved passages (for debugging)")
    for i, doc in enumerate(retrieved, start=1):
        st.write(f"--- Passage {i} (source metadata: {doc.metadata.get('source', 'unknown')}) ---")
        st.write(doc.page_content[:1000])  # show first 1000 chars

