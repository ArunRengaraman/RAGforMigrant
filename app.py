import streamlit as st
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
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

# --------------------
# LLM: HuggingFaceHub FLAN-T5
# --------------------
def get_llm():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

# --------------------
# Load the vectorstore on startup
# --------------------
with st.spinner("Loading knowledge base from trusted websites..."):
    vectordb = prepare_knowledge_base()
    st.success("Knowledge base is ready!")

# --------------------
# Question-answering logic
# --------------------
if user_question:
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

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
