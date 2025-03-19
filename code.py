import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

# ✅ Fix: Ensure `st.set_page_config` is at the top
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

# Environment variables
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize session state variables
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")

    groq_api_key = st.text_input(
        "Groq API Key", 
        value="",
        type="password",
        key="groq_api_key"  # ✅ Fix: Unique key
    )

    groq_model = st.selectbox(
        "Groq Model", 
        ["llama3-70b-8192"], 
        key="groq_model"  # ✅ Fix: Unique key
    )

    temperature = st.slider(
        "Temperature", 
        0.0, 1.0, 0.5, 0.1, 
        key="temperature"  # ✅ Fix: Unique key
    )

    max_tokens = st.slider(
        "Max Tokens", 
        256, 4096, 1024, 256, 
        key="max_tokens"  # ✅ Fix: Unique key
    )

# URL input section
url_col1, url_col2 = st.columns([3, 1])
with url_col1:
    url = st.text_input("🔗 Enter a URL to load content from:", key="url_input")
with url_col2:
    load_button = st.button("Load Content")

# ✅ Now, the keys are unique and should fix the error.

# Rest of the code remains the same.
