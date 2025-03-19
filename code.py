import streamlit as st

st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

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

# ✅ Move this to the first line

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

# Default Groq API key (replace for security)
DEFAULT_GROQ_API_KEY = "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"

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")

    groq_api_key = st.text_input(
        "Groq API Key", 
        value=DEFAULT_GROQ_API_KEY, 
        type="password",
        key="groq_api_key_input"
    )

    groq_model = st.selectbox("Groq Model", ["llama3-70b-8192"], key="groq_model_select")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1, key="temperature_slider")
    max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 256, key="max_tokens_slider")

# URL input section
url_col1, url_col2 = st.columns([3, 1])
with url_col1:
    url = st.text_input("🔗 Enter a URL to load content from:", key="url_input")
with url_col2:
    load_button = st.button("Load Content")

# Functions
def load_content(url):
    with st.spinner(f"📥 Loading content from {url}..."):
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            st.success(f"✅ Successfully loaded content from {url}")
            return documents
        except Exception as e:
            st.error(f"❌ Error loading content: {e}")
            return None

def create_embeddings(documents):
    with st.spinner("🔄 Creating embeddings and vector store..."):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", show_progress=False)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.success("✅ Vector store created successfully")
            return vectorstore.as_retriever()
        except Exception as e:
            st.error(f"❌ Error creating embeddings: {e}")
            return None

def create_question_answering_chain(llm):
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question concisely.\n\n{context}"  
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(llm, qa_prompt)

def initialize_rag_system():
    llm = ChatGroq(
        api_key=groq_api_key,
        model=groq_model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    st.sidebar.success(f"✅ Using Groq LLM: {groq_model}")

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant using retrieved knowledge to answer user queries."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt) 
    question_answer_chain = create_question_answering_chain(llm)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Load content when button is clicked
if load_button and url:
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    documents = load_content(url)
    if documents:
        st.session_state.loaded_url = url
        st.session_state.retriever = create_embeddings(documents)
        
        if st.session_state.retriever:
            st.session_state.rag_chain = initialize_rag_system()
            if st.session_state.rag_chain:
                st.success("🚀 RAG system initialized and ready to use!")

# Display loaded URL status
if st.session_state.loaded_url:
    st.info(f"📌 Currently using content from: {st.session_state.loaded_url}")

# Chat interface
st.header("💬 Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if user_input := st.chat_input("Ask a question about the loaded content..."):
    if not st.session_state.loaded_url:
        st.error("⚠️ Please load a URL first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        user_message = HumanMessage(content=user_input)
        st.session_state.chat_history.add_message(user_message)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag_chain.invoke({
                        "input": user_input, 
                        "chat_history": st.session_state.chat_history.messages
                    })
                    response = result.get('answer', "I don't know.")
                    
                    st.write(response)
                    
                    bot_message = AIMessage(content=response)
                    st.session_state.chat_history.add_message(bot_message)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()
    st.success("✅ Chat history cleared!")
