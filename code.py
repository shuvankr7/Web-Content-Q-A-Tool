import streamlit as st
from langchain_anthropic import ChatAnthropic
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

# App title
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
st.title("RAG Chat Assistant")

# Define constants and initialize session state
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_history" not in st.session_state:
    st.session_state.session_history = ChatMessageHistory()

# Sidebar for API keys and model selection
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_choice = st.radio(
        "Select LLM Provider",
        ["Claude (Anthropic)", "Groq"]
    )
    
    # API Keys based on model choice
    if model_choice == "Claude (Anthropic)":
        anthropic_api_key = st.text_input("Anthropic API Key", 
                                         value="", 
                                         type="password",
                                         help="Enter your Anthropic API key")
        
        claude_model = st.selectbox(
            "Claude Model",
            ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        )
    else:
        groq_api_key = st.text_input("Groq API Key", 
                                    value="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", 
                                    type="password")
        
        groq_model = st.selectbox(
            "Groq Model",
            ["llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"]
        )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=256, max_value=4096, value=1024, step=256)

# URL input and loading
url_col1, url_col2 = st.columns([3, 1])
with url_col1:
    url = st.text_input("Enter a URL to load content from:", key="url_input")
with url_col2:
    load_button = st.button("Load Content")

# Functions
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
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            
            # Create FAISS vector store instead of Chroma
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
        "answer concise."
        "\n\n"
        "{context}"  
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),  
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)

def initialize_rag_system():
    # Create LLM based on user choice
    if model_choice == "Claude (Anthropic)":
        # Check if Anthropic API key is provided
        if not anthropic_api_key:
            st.sidebar.error("Please enter your Anthropic API key")
            return None
            
        # Initialize Claude LLM
        llm = ChatAnthropic(
            model=claude_model, 
            api_key=anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        st.sidebar.success(f"Using Claude LLM: {claude_model}")
    else:
        # Initialize Groq LLM
        llm = ChatGroq(
            api_key=groq_api_key,
            model=groq_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        st.sidebar.success(f"Using Groq LLM: {groq_model}")

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

    # Create history-aware retriever and question-answering chain
    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt) 
    question_answer_chain = create_question_answering_chain(llm)

    # Create RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Logic for loading URL
if load_button and url:
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    documents = load_content(url)
    if documents:
        st.session_state.loaded_url = url
        st.session_state.retriever = create_embeddings(documents)
        
        if st.session_state.retriever:
            # Check if API key is provided when using Anthropic
            if model_choice == "Claude (Anthropic)" and not anthropic_api_key:
                st.sidebar.error("Please enter your Anthropic API key to initialize the RAG system")
            else:
                st.session_state.rag_chain = initialize_rag_system()
                if st.session_state.rag_chain:
                    st.success("RAG system initialized and ready to use!")

# Display loaded URL status
if st.session_state.loaded_url:
    st.info(f"Currently using content from: {st.session_state.loaded_url}")

# Chat interface
st.header("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if user_input := st.chat_input("Ask a question about the loaded content..."):
    # Don't allow input until URL is loaded
    if not st.session_state.loaded_url:
        st.error("Please load a URL first.")
    # Check if API key is provided when using Anthropic
    elif model_choice == "Claude (Anthropic)" and not anthropic_api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    elif not st.session_state.rag_chain:
        st.error("RAG system is not initialized. Please check your configuration.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to session history for RAG
        user_message = HumanMessage(content=user_input)
        st.session_state.session_history.add_message(user_message)
        
        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve context-aware response
                    result = st.session_state.rag_chain.invoke({
                        "input": user_input, 
                        "chat_history": st.session_state.session_history.messages
                    })
                    
                    # Extract answer from result
                    if 'answer' in result:
                        response = result['answer']
                    elif 'output' in result:
                        response = result['output']
                    else:
                        # If neither key exists, get the first value as a fallback
                        response = next(iter(result.values()))
                    
                    # Display the response
                    st.write(response)
                    
                    # Add bot response to session history for RAG
                    bot_message = AIMessage(content=response)
                    st.session_state.session_history.add_message(bot_message)
                    
                    # Add bot response to chat history for display
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.write(f"Error details: {str(e)}")
                    if 'result' in locals():
                        st.json(result)

# Add clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.session_history = ChatMessageHistory()
    st.success("Chat history cleared!")
