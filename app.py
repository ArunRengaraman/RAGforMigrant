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
top_k = st.sidebar.slider("Top K Documents", 1, 10, 3)
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
    You are a helpful assistant for migrants seeking information. Use the provided context to answer questions accurately and helpfully.
    
    IMPORTANT INSTRUCTIONS:
    1. Answer the question using ONLY the information provided in the context below
    2. If the context contains relevant information, provide a detailed and helpful answer
    3. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question completely."
    4. Be specific and cite relevant details from the context when possible
    5. If you find partial information, share what you know and mention what's missing

    Context from documents:
    {context}
    
    Question: {input}
    
    Answer:
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
        
        # Check if we have uploaded documents, otherwise use directory
        if 'uploaded_docs' in st.session_state:
            st.session_state.docs = st.session_state.uploaded_docs
            st.write("📁 Using uploaded documents")
        else:
            st.session_state.loader = PyPDFDirectoryLoader('RAGforMigrant/data')
            st.session_state.docs = st.session_state.loader.load()
            st.write("📁 Using documents from RAGforMigrant/data directory")
        
        # Debug: Show loaded documents info
        st.write(f"📄 Loaded {len(st.session_state.docs)} documents")
        if st.session_state.docs:
            st.write(f"📝 First document preview: {st.session_state.docs[0].page_content[:200]}...")
        else:
            st.error("❌ No documents found in 'RAGforMigrant/data' directory!")
            return False
        
        # Step 3: Split documents
        st.write("✂️ Splitting documents into chunks...")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:200]
        )
        
        # Debug: Show chunking results
        st.write(f"🧩 Created {len(st.session_state.final_documents)} document chunks")
        if st.session_state.final_documents:
            st.write(f"📝 Sample chunk: {st.session_state.final_documents[0].page_content[:200]}...")
        
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
        
        # Test the vector store immediately
        st.write("🧪 Testing vector store...")
        test_results = st.session_state.vectors.similarity_search("test query", k=1)
        st.write(f"✅ Vector store test successful - found {len(test_results)} results")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during vector embedding: {str(e)}")
        st.error(f"📍 Error details: {type(e).__name__}")
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
                # Step 1: Test similarity search first
                st.write("🔍 Performing similarity search...")
                similar_docs = st.session_state.vectors.similarity_search(user_input, k=top_k)
                
                if not similar_docs:
                    st.error("❌ No similar documents found. This might indicate an issue with the vector store.")
                    st.info("💡 Try re-embedding the documents or check if your question relates to the document content.")
                else:
                    st.write(f"✅ Found {len(similar_docs)} relevant document chunks")
                    
                    # Debug: Show what was retrieved
                    with st.expander("🔍 Debug: Retrieved Document Chunks"):
                        for i, doc in enumerate(similar_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(f"Content preview: {doc.page_content[:300]}...")
                            st.write(f"Metadata: {doc.metadata}")
                            st.markdown("---")
                
                # Step 2: Create and run the RAG chain
                document_chain = create_stuff_documents_chain(groq_llm(), prompt)
                retriever = st.session_state.vectors.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k}
                )
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

                # Similarity Search Results
                with st.expander("📚 Document Sources Used"):
                    context_docs = response.get("context", [])
                    if context_docs:
                        for i, doc in enumerate(context_docs, start=1):
                            st.markdown(f"**Source Document {i}:**")
                            st.write(doc.page_content)
                            st.caption(f"Metadata: {doc.metadata}")
                            st.markdown("---")
                    else:
                        st.warning("⚠️ No context documents were used in generating the answer.")
                
                # Debug: Show retrieval chain response structure
                with st.expander("🛠️ Debug: Full Response Structure"):
                    st.json({
                        "input": response.get("input", ""),
                        "context_length": len(response.get("context", [])),
                        "answer_length": len(answer),
                        "has_context": bool(response.get("context"))
                    })
                        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.error(f"📍 Error type: {type(e).__name__}")
            st.info("💡 Try re-embedding the documents if the error persists.")
            
            # Debug information
            with st.expander("🐛 Debug Information"):
                st.write(f"Session state keys: {list(st.session_state.keys())}")
                st.write(f"Vectors object exists: {'vectors' in st.session_state}")
                if 'vectors' in st.session_state:
                    st.write(f"Vectors is None: {st.session_state.vectors is None}")


# ---- Q&A History ----
if st.session_state.history:
    st.subheader("📝 Conversation History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")


# ---- Debug Panel ----
if st.sidebar.checkbox("🐛 Show Debug Information"):
    st.subheader("🔧 Debug Panel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Session State:**")
        st.write(f"- Documents loaded: {'docs' in st.session_state}")
        st.write(f"- Final documents: {'final_documents' in st.session_state}")
        st.write(f"- Embeddings ready: {'embeddings' in st.session_state}")
        st.write(f"- Vector store ready: {'vectors' in st.session_state and st.session_state.vectors is not None}")
        
        if 'docs' in st.session_state:
            st.write(f"- Document count: {len(st.session_state.docs)}")
        if 'final_documents' in st.session_state:
            st.write(f"- Chunk count: {len(st.session_state.final_documents)}")
    
    with col2:
        st.markdown("**Test Vector Store:**")
        if 'vectors' in st.session_state and st.session_state.vectors is not None:
            test_query = st.text_input("Test query:", value="migration")
            if st.button("🧪 Test Similarity Search") and test_query:
                try:
                    test_results = st.session_state.vectors.similarity_search(test_query, k=3)
                    st.write(f"Found {len(test_results)} results")
                    for i, doc in enumerate(test_results):
                        st.write(f"**Result {i+1}:** {doc.page_content[:100]}...")
                except Exception as e:
                    st.error(f"Test failed: {e}")
        else:
            st.write("Vector store not available")
    
    # Show sample documents
    if 'final_documents' in st.session_state and st.session_state.final_documents:
        st.markdown("**Sample Document Chunks:**")
        sample_count = min(3, len(st.session_state.final_documents))
        for i in range(sample_count):
            with st.expander(f"Sample Chunk {i+1}"):
                st.write(st.session_state.final_documents[i].page_content)
                st.caption(f"Metadata: {st.session_state.final_documents[i].metadata}")


# ---- Document Upload Handler ----
if upload_docs:
    st.subheader("📁 Document Upload")
    
    # Create temporary directory for uploaded files
    temp_dir = tempfile.mkdtemp()
    uploaded_files = []
    
    for uploaded_file in upload_docs:
        # Save uploaded file to temp directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        uploaded_files.append(file_path)
        st.write(f"✅ Uploaded: {uploaded_file.name}")
    
    if st.button("📥 Process Uploaded Documents"):
        try:
            # Load uploaded PDFs
            docs = []
            for file_path in uploaded_files:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            
            st.write(f"📄 Loaded {len(docs)} pages from uploaded documents")
            
            # Store in session state for embedding
            st.session_state.uploaded_docs = docs
            st.success("✅ Documents processed. Click 'Embed Documents' to create the vector database.")
            
        except Exception as e:
            st.error(f"❌ Error processing uploads: {e}")
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---- Cleanup on app restart/stop ----
def cleanup_on_exit():
    """Cleanup function to be called when the app stops."""
    safely_close_objectbox_store()

# Register cleanup function
import atexit
atexit.register(cleanup_on_exit)
