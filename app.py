import streamlit as st
import time
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Migrant Resource Assistant", layout="wide")
st.title("🧭 RAG-based Resource Assistant for Migrant Families")
user_question = st.text_input("❓ Ask a question about resources, support, or policies:")

# --------------------
# Predefined URLs to scrape
# --------------------
PREDEFINED_URLS = [
    "https://www.acf.hhs.gov/orr",  # Office of Refugee Resettlement
    "https://www.uscis.gov/humanitarian/refugees-and-asylees",
    "https://www.usa.gov/immigration-and-citizenship",
    "https://www.womensrefugeecommission.org/",
    "https://www.unhcr.org/en-us/"
]

# --------------------
# Load Embeddings (cached)
# --------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------
# Load and embed website content (cached)
# --------------------
@st.cache_resource
def prepare_knowledge_base():
    try:
        loader = WebBaseLoader(PREDEFINED_URLS)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(raw_docs)
        embeddings = load_embeddings()
        vectordb = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        return vectordb
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return None

# --------------------
# LLM: HuggingFace Hub
# --------------------
@st.cache_resource
def get_llm():
    try:
        # Make sure the API token is set
        if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
            st.error("Please set HUGGINGFACEHUB_API_TOKEN in your Streamlit secrets")
            return None
            
        # Set the environment variable for HuggingFace authentication
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        
        # Use HuggingFaceHub with minimal parameters
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5}
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# --------------------
# Load the vectorstore on startup
# --------------------
if "vectordb" not in st.session_state:
    with st.spinner("Loading knowledge base from trusted websites..."):
        st.session_state.vectordb = prepare_knowledge_base()
        if st.session_state.vectordb:
            st.success("Knowledge base is ready!")
        else:
            st.error("Failed to load knowledge base")

# --------------------
# Question-answering logic
# --------------------
if user_question and st.session_state.get("vectordb"):
    try:
        st.info("Initializing LLM...")
        llm = get_llm()
        if llm:
            st.info("LLM initialized successfully. Setting up QA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vectordb.as_retriever(),
                return_source_documents=True
            )
            
            st.info("Processing your question...")
            start = time.time()
            result = qa_chain({"query": user_question})
            end = time.time()
            
            st.subheader("📢 Answer")
            st.write(result["result"])
            st.caption(f"⏱️ Response Time: {end - start:.2f} seconds")
            
            with st.expander("📄 Retrieved Document Snippets"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'unknown')}")
                    st.write(doc.page_content[:500] + "...")
                    st.markdown("---")
        else:
            st.error("Failed to initialize LLM. Please check your configuration.")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        st.info("Please try rephrasing your question or check if the knowledge base is loaded correctly.")
