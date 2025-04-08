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
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default Groq API key
DEFAULT_GROQ_API_KEY = "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"

# Initialize session state
def initialize_session_state():
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

# Load content from URL
def load_content(url):
    with st.spinner(f"Loading content from {url}..."):
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            st.success(f"Successfully loaded content from {url}")
            return documents
        except Exception as e:
            st.error(f"Error loading content: {str(e)}")
            return None

# Create embeddings and vector store
def create_embeddings(documents):
    with st.spinner("Creating embeddings and vector store..."):
        try:
            # Explicitly import torch to ensure it’s fully loaded
            import torch
            # Use a lightweight model and CPU-only to avoid GPU-related issues
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model=sentence_model)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.success("Vector store created successfully")
            return vectorstore.as_retriever()
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return None

# Create question-answering chain
def create_question_answering_chain(llm):
    system_prompt = (
        "You are a precise question-answering assistant. "
        "Answer the user's question using only the context provided from the scraped URL content and the chat history. "
        "Do not use external knowledge beyond what is retrieved. "
        "Keep your answer concise, limited to three sentences maximum, and say 'I don’t know' if the context lacks the answer.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(llm, qa_prompt)

# Initialize RAG system
def initialize_rag_system(groq_api_key, groq_model, temperature, max_tokens):
    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model=groq_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        st.sidebar.success(f"Using Groq LLM: {groq_model}")

        contextualize_q_system_prompt = (
            "You are a precise question-answering assistant. "
            "Rephrase the user's question to incorporate context from the scraped URL content and chat history, "
            "ensuring it relies only on this information and not external knowledge. "
            "Keep the rephrased question clear and concise.\n\nQuestion: {input}"
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
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Main app
def main():
    st.set_page_config(page_title="WebPage Query", layout="wide")
    initialize_session_state()

    # Display Streamlit version for debugging
    st.write(f"Streamlit version: {st.__version__}")
    groq_api_key = DEFAULT_GROQ_API_KEY
    groq_model = "llama3-70b-8192"
    max_tokens= 1024
    # Sidebar configuration
    with st.sidebar:
        st.sidebar.header("Configuration")
        # groq_api_key = st.text_input(
        #     "Groq API Key",
        #     value=DEFAULT_GROQ_API_KEY,
        #     type="password",
        #     help="Use the provided key or enter your own",
        #     key="groq_api_key_input_unique"
        # )
        # groq_model = st.selectbox(
        #     "Groq Model",
        #     ["llama3-70b-8192"],
        #     key="groq_model_select_unique"
        # )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="temperature_slider_unique"
        )
        # max_tokens = st.slider(
        #     "Max Tokens",
        #     min_value=256,
        #     max_value=4096,
        #     value=1024,
        #     step=256,
        #     key="max_tokens_slider_unique"
        # )

    # URL input section
    st.header("Load Content")
    url_col1, url_col2 = st.columns([3, 1])
    with url_col1:
        url = st.text_input("Enter a URL to load content from:", key="url_input_unique")
    with url_col2:
        load_button = st.button("Load Content", key="load_content_button_unique")

    # Handle URL loading
    if load_button and url:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        documents = load_content(url)
        if documents:
            st.session_state.loaded_url = url
            st.session_state.retriever = create_embeddings(documents)
            if st.session_state.retriever:
                st.session_state.rag_chain = initialize_rag_system(groq_api_key, groq_model, temperature, max_tokens)
                if st.session_state.rag_chain:
                    st.success("RAG system initialized and ready to use!")

    # Display loaded URL status
    if st.session_state.loaded_url:
        st.info(f"Currently using content from: {st.session_state.loaded_url}")

    # Chat interface
    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle chat input
    if user_input := st.chat_input("Ask a question about the loaded content...", key="chat_input_unique"):
        if not st.session_state.loaded_url:
            st.error("Please load a URL first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            user_message = HumanMessage(content=user_input)
            st.session_state.chat_history.add_message(user_message)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_chain.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history.messages
                        })
                        response = result.get('answer') or result.get('output') or next(iter(result.values()), "I don't know.")
                        st.write(response)
                        bot_message = AIMessage(content=response)
                        st.session_state.chat_history.add_message(bot_message)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

    # Clear chat button
    if st.button("Clear Chat", key="clear_chat_button_unique"):
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.success("Chat history cleared!")

if __name__ == "__main__":
    main()
