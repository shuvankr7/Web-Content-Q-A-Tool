st.title("Web Content Conversational Q&A Tool (Free Version)")
st.write("Enter URLs, scrape content, and ask questions based on the content.")

# Input URLs
urls = st.text_area("Enter URLs (one per line):")
process_button = st.button("Process URLs")

# Initialize storage for scraped content
texts = []

if process_button:
    urls = urls.split("\n")
    st.write("Scraping and processing content...")
    
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join([p.get_text() for p in soup.find_all("p")])
            texts.append(text)
        except Exception as e:
            st.write(f"Error scraping {url}: {e}")
    
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents(texts)
    
    # Embedding & Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Load HuggingFace Model for Q&A
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    
    # Initialize Memory for Conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize Conversational Q&A Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    st.session_state["qa"] = qa
    st.success("Content processed successfully! You can now ask questions.")

# Chat Interface
if "qa" in st.session_state:
    qa = st.session_state["qa"]
    chat_history = st.session_state.get("chat_history", [])
    
    question = st.text_input("Ask a question based on the processed content:")
    ask_button = st.button("Send")
    
    if ask_button and question:
        response = qa.run(question)
        chat_history.append((question, response))
        st.session_state["chat_history"] = chat_history
        
    st.write("### Chat History")
    for q, a in chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")
