# import all necessary libraries
import time
import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from utils import groq_llm, huggingface_instruct_embedding
import shutil
import tempfile


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

if st.sidebar.button("🗑 Clear Embeddings"):
    st.session_state.clear()
    st.sidebar.success("Embeddings cleared. Please re-embed documents.")

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



def vector_embedding():
    # Close any previously opened store
    if 'vectors' in st.session_state and st.session_state.vectors is not None:
        try:
            st.session_state.vectors.close()  # Close the ObjectBox store
        except Exception as e:
            st.warning(f"⚠️ Failed to close previous ObjectBox store: {e}")
        finally:
            st.session_state.vectors = None

    # Prepare embeddings and load PDFs
    st.session_state.embeddings = huggingface_instruct_embedding()
    st.session_state.loader = PyPDFDirectoryLoader('RAGforMigrant/data')
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:200]
    )

    # Persistent path inside the Git repo
    repo_root = os.path.dirname(os.path.abspath(__file__))  # path to app.py
    db_path = os.path.join(repo_root, "objectbox")

    # Remove old DB directory to avoid schema/lock errors
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)
    os.makedirs(db_path, exist_ok=True)

    # Create new ObjectBox store
    st.session_state.vectors = ObjectBox.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings,
        embedding_dimensions=768,
        db_directory=db_path
    )

    st.write(f"✅ ObjectBox DB created at: {db_path}")



# ---- Embedding Trigger ----
if st.sidebar.button('📥 Embed Documents'):
    vector_embedding()
    st.sidebar.success('✅ Database is ready. You can now enter your question.')

# ---- User Query ----
user_input = st.text_input('💬 Enter your question from documents')

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Processing Query ----
if user_input:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click **'Embed Documents'** first to prepare the database.")
    else:
        document_chain = create_stuff_documents_chain(groq_llm(), prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()

        response = retrieval_chain.invoke({'input': user_input})
        answer = response['answer']
        st.session_state.history.append((user_input, answer))
        
        st.success(answer)
        st.caption(f"⏱ Response time: {(time.process_time() - start):.2f} secs")

        # Similarity Search
        with st.expander("📚 Document Similarity Search"):
            for i, doc in enumerate(response["context"], start=1):
                st.markdown(f"**Document {i}:**")
                st.write(doc.page_content)
                st.caption(f"Source: {doc.metadata}")

# ---- Q&A History ----
if st.session_state.history:
    st.subheader("📝 Conversation History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
