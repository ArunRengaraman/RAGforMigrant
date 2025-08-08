# app.py - Updated RAGforMigrant with safe ObjectBox init (persistent repo storage)

# import all necessary libraries
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

# ---- Streamlit Page Config ----
st.set_page_config(layout='wide', page_title="RAG for Migrants", page_icon="🌍")
st.title('🌍 RAG for Migrants')
st.markdown("Empowering migrants with information retrieval using **ObjectBox** and **LangChain**")

# ---- Sidebar ----
st.sidebar.header("⚙️ Configuration")
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, step=50)
top_k = st.sidebar.slider("Top K Documents", 1, 10, 3)
upload_docs = st.sidebar.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# persistent repo objectbox folder (path relative to this script)
repo_root = os.path.dirname(os.path.abspath(__file__))
REPO_OBJECTBOX_DIR = os.path.join(repo_root, "objectbox")

# helper: ensure repo objectbox folder exists (do not delete here!)
if not os.path.exists(REPO_OBJECTBOX_DIR):
    try:
        os.makedirs(REPO_OBJECTBOX_DIR, exist_ok=True)
    except Exception as e:
        st.error(f"Could not create objectbox directory at {REPO_OBJECTBOX_DIR}: {e}")

# ---- Prompt Template ----
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

# ---- Utility functions ----
def safe_remove_folder(path):
    """Remove a folder if it exists. Return True if removed or not present."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            return True
        except Exception as e:
            st.error(f"Failed to remove folder {path}: {e}")
            return False
    return True

def close_vectors():
    """
    Try to dereference/close vectorstore in session_state so a new store can be opened.
    We can't always call a close() API on the langchain wrapper; removing the reference
    and letting python GC usually works. If ObjectBox exposes an internal ._db attribute,
    we try to close it safely.
    """
    if "vectors" in st.session_state:
        vec = st.session_state.pop("vectors", None)
        # attempt to close underlying DB if available
        try:
            db = getattr(vec, "_db", None)
            if db is not None:
                # objectbox store likely has 'close' method
                try:
                    db.close()
                except Exception:
                    pass
        except Exception:
            pass

# ---- Clear Embeddings Button (in sidebar) ----
if st.sidebar.button("🗑 Clear Embeddings (delete DB)"):
    # close any open vectorstore in memory, then delete folder on disk
    close_vectors()
    removed = safe_remove_folder(REPO_OBJECTBOX_DIR)
    if removed:
        # recreate empty folder so path exists for later
        try:
            os.makedirs(REPO_OBJECTBOX_DIR, exist_ok=True)
            st.sidebar.success("Embeddings cleared and DB folder deleted. Re-run embedding to recreate.")
        except Exception as e:
            st.sidebar.error(f"Deleted but failed to recreate folder: {e}")

# ---- Initialization: embeddings + try to load existing ObjectBox once ----
# We initialize embeddings once and attempt to load existing ObjectBox store (if present).
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    # create embeddings (this may be expensive)
    try:
        st.session_state.embeddings = huggingface_instruct_embedding()
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {e}")
        st.stop()

    # If there appears to be an existing objectbox DB, try to open it once and set vectors
    # We wrap this in try/except to avoid the app crashing.
    try:
        # Only attempt to load if the folder contains files
        if any(os.scandir(REPO_OBJECTBOX_DIR)):
            try:
                # Attempt to instantiate ObjectBox with the embedding to open the DB
                st.session_state.vectors = ObjectBox(st.session_state.embeddings, db_directory=REPO_OBJECTBOX_DIR)
                st.success(f"Loaded existing ObjectBox DB from: {REPO_OBJECTBOX_DIR}")
            except Exception as e_open:
                # If we can't open, warn but do not crash; user can re-embed to rebuild DB.
                st.warning("Found an objectbox DB folder in the repo but could not open it automatically. "
                           "You can re-create it by clicking 'Embed Documents'.")
                st.info(f"Open error (non-fatal): {e_open}")
        else:
            # folder empty → nothing to load yet
            pass
    except Exception:
        # any unexpected error - keep going; user can click Embed Documents to create db
        pass

# ---- vector embedding function (creates/rebuilds DB) ----
def vector_embedding(rebuild=False):
    """
    Create embeddings and build ObjectBox DB from PDF documents (and uploaded PDFs).
    Set st.session_state.vectors to the created ObjectBox vectorstore.
    If rebuild=True, force deletion of any existing DB first.
    """
    # If vectors already set and not rebuilding, do nothing
    if "vectors" in st.session_state and not rebuild:
        st.info("Vectorstore already exists in session. If you want to rebuild, click 'Rebuild DB'.")
        return

    # If rebuild requested, close and delete old DB
    if rebuild:
        close_vectors()
        safe_remove_folder(REPO_OBJECTBOX_DIR)
        os.makedirs(REPO_OBJECTBOX_DIR, exist_ok=True)

    # Load documents (uploaded PDFs take precedence)
    docs = []
    try:
        if upload_docs:
            # user uploaded files
            for file in upload_docs:
                loader = PyPDFLoader(file)
                docs.extend(loader.load())
        else:
            # use local repo data folder
            loader = PyPDFDirectoryLoader('RAGforMigrant/data')
            docs = loader.load()
    except Exception as e:
        st.error(f"Failed to load PDFs: {e}")
        return

    if not docs:
        st.warning("No documents found to embed. Upload PDFs or add them to 'RAGforMigrant/data'.")
        return

    # split docs into chunks with user-selected chunk size/overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_docs = text_splitter.split_documents(docs)

    # Build the ObjectBox DB (this will create files under REPO_OBJECTBOX_DIR)
    try:
        # If a vectors object exists in-memory, close first (rare)
        close_vectors()
        st.session_state.vectors = ObjectBox.from_documents(
            final_docs,
            st.session_state.embeddings,
            embedding_dimensions=768,  # ensure this matches your embedding model output
            db_directory=REPO_OBJECTBOX_DIR
        )
        st.success(f"✅ ObjectBox DB created at: {REPO_OBJECTBOX_DIR}")
    except Exception as e:
        st.error("Failed to create or open ObjectBox DB. See details below.")
        st.exception(e)
        # If failed, try to tidy up partial DB
        try:
            close_vectors()
            safe_remove_folder(REPO_OBJECTBOX_DIR)
            os.makedirs(REPO_OBJECTBOX_DIR, exist_ok=True)
        except Exception:
            pass

# ---- Embedding Trigger Buttons ----
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button('📥 Embed Documents'):
        vector_embedding(rebuild=False)
with col2:
    if st.button('🔁 Rebuild DB (force)'):
        vector_embedding(rebuild=True)

# ---- User Query ----
user_input = st.text_input('💬 Enter your question from documents')

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Processing Query ----
if user_input:
    if "vectors" not in st.session_state:
        st.warning("⚠️ No vectorstore available. Click 'Embed Documents' to create the DB first.")
    else:
        try:
            document_chain = create_stuff_documents_chain(groq_llm(), prompt)
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()

            response = retrieval_chain.invoke({'input': user_input})
            answer = response.get('answer', "(no answer returned)")
            st.session_state.history.append((user_input, answer))

            st.success(answer)
            st.caption(f"⏱ Response time: {(time.process_time() - start):.2f} secs")

            # Similarity Search
            with st.expander("📚 Document Similarity Search"):
                ctx = response.get("context", [])
                if not ctx:
                    st.write("No context documents returned.")
                for i, doc in enumerate(ctx, start=1):
                    st.markdown(f"**Document {i}:**")
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata}")
        except Exception as e:
            st.error("Error during retrieval chain execution.")
            st.exception(e)

# ---- Q&A History ----
if st.session_state.history:
    st.subheader("📝 Conversation History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
