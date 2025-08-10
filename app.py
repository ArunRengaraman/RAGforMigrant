# app.py - Updated RAGforMigrant with robust embedding and ObjectBox handling

import time
import streamlit as st
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from utils import groq_llm, huggingface_instruct_embedding
from typing import Optional

# ---- Page config ----
st.set_page_config(layout="wide", page_title="RAG for Migrants", page_icon="🌍")
st.title("🌍 RAG for Migrants")
st.markdown("Empowering migrants with information retrieval using **ObjectBox** and **LangChain**")

# ---- Sidebar controls ----
st.sidebar.header("⚙️ Configuration")
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, step=50)
top_k = st.sidebar.slider("Top K Documents", 1, 10, 3)
upload_docs = st.sidebar.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# repo-level objectbox path
repo_root = os.path.dirname(os.path.abspath(__file__))
REPO_OBJECTBOX_DIR = os.path.join(repo_root, "objectbox")

# ensure directory exists
if not os.path.exists(REPO_OBJECTBOX_DIR):
    try:
        os.makedirs(REPO_OBJECTBOX_DIR, exist_ok=True)
    except Exception as e:
        st.error(f"Could not create objectbox directory at {REPO_OBJECTBOX_DIR}: {e}")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>
    Question: {input}
    """
)

# ---- Helpers for ObjectBox safe handling ----
def close_vectors_safe():
    """Try to close and remove vectorstore from session_state cleanly."""
    if "vectors" in st.session_state:
        vec = st.session_state.pop("vectors", None)
        try:
            db = getattr(vec, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass
        except Exception:
            pass

def remove_repo_db_folder():
    """Delete the repo objectbox folder (use with caution)."""
    try:
        if os.path.exists(REPO_OBJECTBOX_DIR):
            shutil.rmtree(REPO_OBJECTBOX_DIR)
        os.makedirs(REPO_OBJECTBOX_DIR, exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Failed to remove objectbox folder: {e}")
        return False

# ---- Embeddings initialization (once) ----
@st.cache_resource
def get_embeddings_cached():
    """
    Load and return embeddings. Cached resource ensures it's loaded once per app instance.
    """
    return huggingface_instruct_embedding()

try:
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = get_embeddings_cached()
except Exception as e:
    st.error("Failed to initialize embeddings. Details:")
    st.exception(e)
    st.stop()

# ---- Buttons for clearing/rebuilding DB ----
if st.sidebar.button("🗑 Clear DB (delete files)"):
    close_vectors_safe()
    removed = remove_repo_db_folder()
    if removed:
        st.sidebar.success("DB folder deleted. You can re-embed to recreate the DB.")

# ---- vector embedding function ----
def vector_embedding(rebuild: bool = False) -> Optional[str]:
    """
    Build ObjectBox DB from PDFs (uploaded or in repo/data).
    If rebuild=True, force deletion first.
    Returns path if successful, else None.
    """
    # If vectors already exist and not rebuilding, skip
    if "vectors" in st.session_state and not rebuild:
        st.info("Vectorstore already loaded in session. Use Rebuild to recreate.")
        return REPO_OBJECTBOX_DIR

    # If rebuild requested, close and remove existing
    if rebuild:
        close_vectors_safe()
        remove_repo_db_folder()

    # Load documents (uploaded PDFs take precedence)
    docs = []
    try:
        if upload_docs:
            for f in upload_docs:
                loader = PyPDFLoader(f)
                docs.extend(loader.load())
        else:
            loader = PyPDFDirectoryLoader("RAGforMigrant/data")
            docs = loader.load()
    except Exception as e:
        st.error("Failed to load PDF documents.")
        st.exception(e)
        return None

    if not docs:
        st.warning("No documents found. Add PDFs to RAGforMigrant/data or upload via sidebar.")
        return None

    # Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_docs = splitter.split_documents(docs)

    # Ensure folder exists
    os.makedirs(REPO_OBJECTBOX_DIR, exist_ok=True)

    # Build objectbox vectorstore
    try:
        # close prior in-memory store if any
        close_vectors_safe()
        st.session_state.vectors = ObjectBox.from_documents(
            final_docs,
            st.session_state.embeddings,
            embedding_dimensions=768,  # ensure this matches your embedding model
            db_directory=REPO_OBJECTBOX_DIR,
        )
        st.success(f"✅ ObjectBox DB ready at: {REPO_OBJECTBOX_DIR}")
        return REPO_OBJECTBOX_DIR
    except Exception as e:
        st.error("Failed to create or open ObjectBox vectorstore.")
        st.exception(e)
        # attempt cleanup
        try:
            close_vectors_safe()
            remove_repo_db_folder()
        except Exception:
            pass
        return None

# Buttons: embed / rebuild
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("📥 Embed Documents"):
        vector_embedding(rebuild=False)
with col2:
    if st.button("🔁 Rebuild DB"):
        vector_embedding(rebuild=True)

# ---- Query UI ----
user_input = st.text_input("💬 Enter your question from documents")

if "history" not in st.session_state:
    st.session_state.history = []

if user_input:
    if "vectors" not in st.session_state:
        st.warning("No vectorstore available. Click 'Embed Documents' to create it first.")
    else:
        try:
            document_chain = create_stuff_documents_chain(groq_llm(), prompt)
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_input})
            answer = response.get("answer", "(no answer returned)")
            st.session_state.history.append((user_input, answer))
            st.success(answer)
            st.caption(f"⏱ Response time: {(time.process_time() - start):.2f} secs")

            with st.expander("📚 Document Similarity Search"):
                ctx = response.get("context", [])
                if not ctx:
                    st.write("No context documents returned.")
                for i, doc in enumerate(ctx, start=1):
                    st.markdown(f"**Document {i}:**")
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata}")
        except Exception as e:
            st.error("Error while running retrieval chain.")
            st.exception(e)

# ---- History ----
if st.session_state.history:
    st.subheader("📝 Conversation History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
