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
import os

# Page configuration
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

# Set environment variables
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Default configuration
DEFAULT_GROQ_API_KEY = "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"
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024

# Initialize session state - only once
def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.retriever = None
        st.session_state.rag_chain = None
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.session_state.loaded_url = None
        st.session_state.url_input = ""
        st.session_state.groq_api_key = DEFAULT_GROQ_API_KEY
        st.session_state.groq_model = DEFAULT_MODEL
        st.session_state.temperature = DEFAULT_TEMPERATURE
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS

init_session_state()

# Utility functions
def load_content(url):
    """Load content from a URL using WebBaseLoader."""
    with st.spinner(f"Loading content from {url}..."):
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            st.success(f"Successfully loaded content from {url}")
            return documents
        except Exception as e:
            st.error(f"Error loading content: {e}")
            return None

def create_embeddings(documents):
    """Create embeddings for the documents using HuggingFaceEmbeddings and FAISS."""
    with st.spinner("Creating embeddings and vector store..."):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                show_progress=False
            )
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.success("Vector store created successfully")
            return vectorstore.as_retriever()
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            return None

def create_question_answering_chain(llm):
    """Create a question-answering chain using the given LLM."""
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"  
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(llm, qa_prompt)

def initialize_rag_system():
    """Initialize the RAG system with Groq LLM."""
    llm = ChatGroq(
        api_key=st.session_state.groq_api_key,
        model=st.session_state.groq_model,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens
    )
    
    contextualize_q_system_prompt = (
        """You are tasked with answering a question based on three sources of context:
        1. **Website Content:** The text extracted from the user's provided URLs.
        2. **Chat History:** The conversation history between the user and the assistant.
        3. **User Question:** The current question being asked by the user.
        Please answer the following question based only on the above context, and provide a concise, relevant, and clear response.
        Question: {input}"""
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt) 
    question_answer_chain = create_question_answering_chain(llm)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Event handlers
def on_api_key_change():
    st.session_state.groq_api_key = st.session_state.get("api_key_input", DEFAULT_GROQ_API_KEY)

def on_temperature_change():
    st.session_state.temperature = st.session_state.get("temperature_input", DEFAULT_TEMPERATURE)

def on_max_tokens_change():
    st.session_state.max_tokens = st.session_state.get("max_tokens_input", DEFAULT_MAX_TOKENS)

def on_url_change():
    st.session_state.url_input = st.session_state.get("url_input_field", "")

def on_load_content():
    url = st.session_state.url_input
    if url:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        documents = load_content(url)
        if documents:
            st.session_state.loaded_url = url
            st.session_state.retriever = create_embeddings(documents)
            
            if st.session_state.retriever:
                st.session_state.rag_chain = initialize_rag_system()
                st.sidebar.success(f"Using Groq LLM: {st.session_state.groq_model}")
                st.success("RAG system initialized and ready to use!")

def on_clear_chat():
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()

def on_chat_submit():
    user_input = st.session_state.get("chat_input_field", "")
    
    if not user_input:
        return
    
    if not st.session_state.loaded_url:
        st.error("Please load a URL first.")
        return
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    user_message = HumanMessage(content=user_input)
    st.session_state.chat_history.add_message(user_message)
    
    # Process with RAG chain
    try:
        result = st.session_state.rag_chain.invoke({
            "input": user_input, 
            "chat_history": st.session_state.chat_history.messages
        })
        response = result.get('answer') or result.get('output') or next(iter(result.values()), "I don't know.")
        
        # Add assistant message to chat
        bot_message = AIMessage(content=response)
        st.session_state.chat_history.add_message(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error: {e}")

# Layout the app
st.title("RAG-Powered Chat Assistant")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key
    st.text_input(
        "Groq API Key", 
        value=st.session_state.groq_api_key, 
        key="api_key_input",
        type="password",
        help="You can use the provided API key or enter your own",
        on_change=on_api_key_change
    )
    
    # Model (no need for on_change since only one option)
    st.selectbox(
        "Groq Model",
        [DEFAULT_MODEL],
        index=0,
        key="model_selector"
    )
    
    # Temperature
    st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.temperature, 
        step=0.1,
        key="temperature_input",
        on_change=on_temperature_change
    )
    
    # Max Tokens
    st.slider(
        "Max Tokens", 
        min_value=256, 
        max_value=4096, 
        value=st.session_state.max_tokens, 
        step=256,
        key="max_tokens_input",
        on_change=on_max_tokens_change
    )

# URL input and loading
st.header("Load Content")
col1, col2 = st.columns([3, 1])

with col1:
    st.text_input(
        "Enter a URL to load content from:", 
        value=st.session_state.url_input,
        key="url_input_field",
        on_change=on_url_change
    )
    
with col2:
    st.button(
        "Load Content", 
        key="load_button",
        on_click=on_load_content
    )

# Display loaded URL status
if st.session_state.loaded_url:
    st.info(f"Currently using content from: {st.session_state.loaded_url}")

# Chat interface
st.header("Chat")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Handle user input
st.chat_input(
    "Ask a question about the loaded content...", 
    key="chat_input_field",
    on_submit=on_chat_submit
)

# Clear chat button
st.button(
    "Clear Chat", 
    key="clear_button",
    on_click=on_clear_chat
)
