
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
import gc


# ---- Streamlit Page Config ----
st.set_page_config(layout='wide', page_title="RAG for Migrants", page_icon="🌍")
st.title('🌍 RAG for Migrants')
st.markdown("Empowering migrants with information retrieval using **ObjectBox** and **LangChain**")

# ---- Sidebar ----
st.sidebar.header("⚙️ Configuration")
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, step=50)
top_k = st.sidebar.slider("Top K Documents", 1, 10, 5)
upload_docs = st.sidebar.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)


def safely_close_objectbox_store():
    """Safely close ObjectBox store and clean up resources."""
    if 'vectors' in st.session_state and st.session_state.vectors is not None:
        try:
            # Close the ObjectBox store properly
            if hasattr(st.session_state.vectors, '_db') and st.session_state.vectors._db is not None:
                st.session_state.vectors._db.close()
                st.session_state.vectors._db = None
            
            # Clear the vectors reference
            st.session_state.vectors = None
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to ensure resources are freed
            time.sleep(0.1)
            
        except Exception as e:
            st.warning(f"⚠️ Error closing ObjectBox store: {e}")
            # Force clear anyway
            st.session_state.vectors = None


if st.sidebar.button("🗑 Clear Embeddings"):
    safely_close_objectbox_store()
    
    # Clear session state
    for key in list(st.session_state.keys()):
        if key != 'history':  # Preserve conversation history if desired
            del st.session_state[key]
    
    st.sidebar.success("✅ Embeddings cleared. Please re-embed documents.")


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


def get_unique_db_path():
    """Generate a unique database path to avoid conflicts."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    timestamp = str(int(time.time()))
    db_path = os.path.join(repo_root, f"objectbox_{timestamp}")
    return db_path


def vector_embedding():
    """Create vector embeddings with proper ObjectBox store management."""
    try:
        # Step 1: Safely close any existing store
        safely_close_objectbox_store()
        
        # Step 2: Prepare embeddings and load PDFs
        st.write("📚 Loading embeddings and documents...")
        st.session_state.embeddings = huggingface_instruct_embedding()
        st.session_state.loader = PyPDFDirectoryLoader('RAGforMigrant/data')
        st.session_state.docs = st.session_state.loader.load()
        
        # Step 3: Split documents
        st.write("✂️ Splitting documents into chunks...")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:200]
        )
        
        # Step 4: Create unique database path
        db_path = get_unique_db_path()
        
        # Step 5: Clean up any existing database directories
        repo_root = os.path.dirname(os.path.abspath(__file__))
        
        # Remove old objectbox directories
        for item in os.listdir(repo_root):
            if item.startswith('objectbox'):
                old_path = os.path.join(repo_root, item)
                if os.path.isdir(old_path):
                    try:
                        shutil.rmtree(old_path, ignore_errors=True)
                        st.write(f"🗑️ Removed old database: {item}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not remove {item}: {e}")
        
        # Step 6: Create new database directory
        os.makedirs(db_path, exist_ok=True)
        
        # Step 7: Create new ObjectBox store
        st.write("🔄 Creating new ObjectBox vector store...")
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=768,
            db_directory=db_path,
            clear_db=True  # This ensures a clean start
        )
        
        st.write(f"✅ ObjectBox DB created successfully at: {db_path}")
        st.write(f"📊 Embedded {len(st.session_state.final_documents)} document chunks")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during vector embedding: {str(e)}")
        # Clean up on error
        safely_close_objectbox_store()
        return False


# ---- Embedding Trigger ----
if st.sidebar.button('📥 Embed Documents'):
    with st.spinner('Creating embeddings...'):
        success = vector_embedding()
        if success:
            st.sidebar.success('✅ Database is ready. You can now enter your question.')
        else:
            st.sidebar.error('❌ Failed to create database. Please try again.')


# ---- User Query ----
user_input = st.text_input('💬 Enter your question from documents')

if "history" not in st.session_state:
    st.session_state.history = []


# ---- Processing Query ----
if user_input:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.warning("⚠️ Please click **'Embed Documents'** first to prepare the database.")
    else:
        try:
            with st.spinner('Searching for relevant information...'):
                document_chain = create_stuff_documents_chain(groq_llm(), prompt)
                retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_input})
                answer = response['answer']
                processing_time = time.process_time() - start
                
                # Store in history
                st.session_state.history.append((user_input, answer))
                
                # Display results
                st.success(answer)
                st.caption(f"⏱ Response time: {processing_time:.2f} secs")

                # Similarity Search
                with st.expander("📚 Document Similarity Search"):
                    for i, doc in enumerate(response["context"], start=1):
                        st.markdown(f"**Document {i}:**")
                        st.write(doc.page_content)
                        st.caption(f"Source: {doc.metadata}")
                        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.info("💡 Try re-embedding the documents if the error persists.")


# ---- Q&A History ----
if st.session_state.history:
    st.subheader("📝 Conversation History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")


# ---- Cleanup on app restart/stop ----
def cleanup_on_exit():
    """Cleanup function to be called when the app stops."""
    safely_close_objectbox_store()

# Register cleanup function
import atexit
atexit.register(cleanup_on_exit)
