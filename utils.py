from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from config import load_config, get_groq_api

# load app configuration
load_config()

# setup groq LLM
def groq_llm():
    return ChatGroq(
        groq_api_key=get_groq_api(),
        model_name='Llama3-8b-8192'
    )

# setup huggingface_instruct_embedding
def huggingface_instruct_embedding():
    return HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# load and clean content from a URL
def load_url_content(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Optional extra cleanup
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, "html.parser")
        doc.page_content = soup.get_text(separator="\n")

    return docs
