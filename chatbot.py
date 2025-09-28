import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load env file (make sure GROQ_API_KEY is inside .env)
project_root = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

st.title('Chat Bot Demo')
st.sidebar.title('News Research Tool')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_URL_Clicked = st.sidebar.button('Process_URL')
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

if process_URL_Clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text splitter started...")
    docs = text_splitter.split_documents(data)
    # Add source metadata while creating documents
    docs_with_meta = []
    for i, doc in enumerate(docs):
        doc.metadata["source"] = urls[i % len(urls)]  # attach URL to each chunk
        docs_with_meta.append(doc)



    # Create embeddings with Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = FAISS.from_documents(docs_with_meta, embeddings)
    main_placeholder.text("Embedding vector started building...")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectors, f)

# User query input
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20B"
        )

        # Retrieval Chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        # Display answer
        st.header("Answer")
        st.subheader(result.get("answer", "No answer found."))

        # Display sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)  # This will now show the actual URL stored in metadata
