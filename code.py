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

# Set environment variables
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Groq API setup
DEFAULT_GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"

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

# Load webpage content
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

# Create embeddings and retriever
def create_embeddings(documents):
    with st.spinner("Creating embeddings and vector store..."):
        try:
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

# Create the QA chain
def create_question_answering_chain(llm):
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise question-answering assistant. Answer the user's question using only the context provided. Say 'I don't know' if unsure.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    return create_stuff_documents_chain(llm, qa_prompt)

# Initialize RAG system
def initialize_rag_system(groq_api_key, model, temperature, max_tokens):
    try:
        llm = ChatGroq(api_key=groq_api_key, model=model, temperature=temperature, max_tokens=max_tokens)
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the user's question using only chat history and URL content."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)
        question_answer_chain = create_question_answering_chain(llm)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Main app
def main():
    st.set_page_config(page_title="WebPage Query", layout="wide")
    initialize_session_state()

    st.title("🔍 Web RAG Assistant")
    st.write(f"Streamlit version: {st.__version__}")

    # Sidebar config
    with st.sidebar:
        st.header("Configuration")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5, step=0.1)
        max_tokens = 1024

        # Debug info
        st.write("🔍 Debug Info")
        st.write("Retriever Initialized:", st.session_state.retriever is not None)
        st.write("RAG Chain Initialized:", st.session_state.rag_chain is not None)

    # URL loader
    st.header("Load Content")
    url_col1, url_col2 = st.columns([3, 1])
    with url_col1:
        url = st.text_input("Enter a URL to load content from:")
    with url_col2:
        load_button = st.button("Load Content")

    if load_button and url:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        documents = load_content(url)
        if documents:
            st.session_state.loaded_url = url
            st.session_state.retriever = create_embeddings(documents)
            st.write("✅ Retriever created:", st.session_state.retriever is not None)
            if st.session_state.retriever:
                st.session_state.rag_chain = initialize_rag_system(
                    DEFAULT_GROQ_API_KEY,
                    GROQ_MODEL,
                    temperature,
                    max_tokens
                )
                st.write("✅ RAG chain created:", st.session_state.rag_chain is not None)
                if st.session_state.rag_chain:
                    st.success("RAG system initialized and ready!")

    if st.session_state.loaded_url:
        st.info(f"Currently using content from: {st.session_state.loaded_url}")

    # Chat section
    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_input := st.chat_input("Ask a question about the loaded content..."):
        if not st.session_state.loaded_url:
            st.error("Please load a URL first.")
        elif not st.session_state.rag_chain:
            st.error("RAG system not initialized. Please try reloading the content.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.chat_history.add_message(HumanMessage(content=user_input))

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_chain.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history.messages
                        })
                        response = result.get('answer') or result.get('output') or next(iter(result.values()), "I don't know.")
                        st.write(response)
                        st.session_state.chat_history.add_message(AIMessage(content=response))
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.success("Chat history cleared!")

if __name__ == "__main__":
    main()
