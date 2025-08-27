# # app.py (shortened placeholder)
# import streamlit as st
# st.title("Gemini RAG — Streamlit")
# st.write("Run ingest.py first, then ask questions!")
"""
Streamlit front-end for the Gemini RAG app.

- Loads FAISS vectorstore (if present)
- Accepts uploaded PDFs (session-only)
- Runs semantic search and sends context + question to Gemini (via google-generativeai)
- Uses langchain-google-genai for embeddings (when creating/updating index)
"""
# app.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain + FAISS + embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Gemini SDK
import google.generativeai as genai

# Page settings
st.set_page_config(page_title="Gemini RAG — Streamlit", layout="wide")
st.title("🧠 Gemini RAG — Streamlit (Virat)")
# -------- CONFIG ----------
VS_DIR = "vectorstore/faiss_index"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-pro"   # use "gemini-pro" if that's what you have access to
# --------------------------

load_dotenv()

# ---- API Key Handling ----
def get_api_key():
    try:
        return st.secrets["GEMINI"]["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY")

API_KEY = get_api_key()
if not API_KEY:
    st.sidebar.error("❌ Missing GOOGLE_API_KEY. Add it to .env or .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=API_KEY)



# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Retriever top_k", 1, 8, 3)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2)
    add_files = st.file_uploader("Upload PDFs (session only)", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    st.markdown("📌 For persistent docs, put them in `data/` and run `python ingest.py`.")

# ---- Load FAISS vectorstore ----
@st.cache_resource(show_spinner=False)
def load_vectorstore(path: str):
    if not os.path.exists(path):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

vs = load_vectorstore(VS_DIR)

# ---- Handle uploaded PDFs ----
if add_files:
    tmp_dir = tempfile.gettempdir()
    uploaded_docs = []
    for f in add_files:
        tmp_path = os.path.join(tmp_dir, f.name)
        with open(tmp_path, "wb") as out:
            out.write(f.getbuffer())
        loader = PyPDFLoader(tmp_path)
        uploaded_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(uploaded_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    if vs is None:
        vs = FAISS.from_documents(chunks, embeddings)
    else:
        vs.add_documents(chunks)

    st.sidebar.success(f"✅ Added {len(chunks)} chunks from {len(add_files)} file(s).")

if vs is None:
    st.warning("⚠️ No FAISS index found. Put files in `data/` and run `python ingest.py`.")
    st.stop()

# ---- Chat History ----
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    if role == "user":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Assistant:** {text}")

# ---- Query Box ----
query = st.text_input("💬 Ask something about your documents (press Enter):")

if query:
    st.session_state.history.append(("user", query))

    # 1) semantic search
    docs = vs.similarity_search(query, k=top_k)
    context = "\n\n---\n\n".join([d.page_content for d in docs]) or "No retrieved context."

    # 2) Build prompt
    prompt = (
        "You are a helpful RAG assistant. Use ONLY the provided context to answer.\n"
        "If the answer is not contained in the context, say: 'I don’t know from the provided documents.'\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    # 3) Call Gemini API
    try:
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 1024},
        )
        answer = response.text
    except Exception as e:
        answer = f"❌ Error calling Gemini API: {e}"

    # 4) Display
    st.session_state.history.append(("assistant", answer))
    st.markdown(f"**🤖 Assistant:** {answer}")

    # 5) Sources
    with st.expander("📖 Sources (top-k)"):
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "uploaded PDF")
            page = d.metadata.get("page", "-")
            st.markdown(f"**{i}.** `{src}` — page {page}")
