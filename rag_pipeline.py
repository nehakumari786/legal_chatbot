from langchain_ollama import ChatOllama
from vector_database import load_faiss_db

# LLM (DeepSeek R1) for generation
# Make sure you pulled this model in Ollama: `ollama pull deepseek-r1:8b`
LLM_MODEL_NAME = "deepseek-r1:8b"

# Initialize LLM object (no network key required, uses local Ollama daemon)
llm_model = ChatOllama(model=LLM_MODEL_NAME)

def retrieve_docs_from_db(db, query, k=4):
    """
    Retrieve relevant documents from a FAISS DB instance (pass db object).
    """
    return db.similarity_search(query, k=k)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def answer_query_from_docs(documents, query, model=llm_model):
    """
    Given retrieved LangChain documents and a query, call the LLM and return text.

    NEW BEHAVIOR ADDED:
      - If user provides wrong factual info (wrong roll number, wrong ID, wrong date)
        compared to document context, the assistant:
          1) Detects contradiction,
          2) Corrects the fact using context,
          3) Answers using the corrected fact.
      - If info does not exist in context, says:
        "I cannot find this information in the document."
    """

    context = get_context(documents)

    # Enhanced system prompt for contradiction detection + correction
    prompt = f"""
You are an expert legal/educational assistant.

STRICT RULES YOU MUST FOLLOW:
1. Use ONLY the information from the CONTEXT below. Do NOT use outside knowledge.
2. Before answering, CHECK whether the user's question contains any factual claim
   (such as roll numbers, names, dates, case numbers).
3. If the user's factual claim CONTRADICTS the context:
      - First politely CORRECT the user with the proper fact from context.
      - Then answer the question using the corrected information.
4. If the user's fact is NOT found anywhere in the context:
      - Say: "I cannot find this information in the document."
5. Keep answer clear and concise.

---------------- EXAMPLES ----------------
CONTEXT:
Neha — Roll No: 12
Amit — Roll No: 5

USER:
"What is Neha's roll no 11?"

ASSISTANT:
"It seems the query contains an incorrect roll number. According to the document,
Neha's correct roll number is 12. Using that: Neha is listed with Roll No 12."

------------------------------------------

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    # Call Ollama (DeepSeek-R1:8b)
    response = model.invoke(prompt)

    # Support different LangChain response formats
    if hasattr(response, "content"):
        return response.content.strip()

    return str(response).strip()
