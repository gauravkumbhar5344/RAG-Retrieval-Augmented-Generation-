import os
import streamlit as st
# Document loaders - now in langchain-community
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# Text splitter - now in langchain-text-splitters (standalone)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings - now in langchain-huggingface  
from langchain_huggingface import HuggingFaceEmbeddings

# Vectorstore - install chromadb separately
from langchain_community.vectorstores import Chroma

# RetrievalQA chain - now in langchain-classic (legacy chains)
from langchain_classic.chains import RetrievalQA

# Groq LLM - separate integration package (unchanged)
from langchain_groq import ChatGroq

# Standard library (unchanged)
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 RAG Chatbot using LangChain + Groq + HuggingFace")

# Sidebar
st.sidebar.header("Upload your document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])

if uploaded_file:
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load documents based on file type
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(tmp_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(tmp_path)
    else:
        st.error("Unsupported file type!")
        st.stop()

    st.sidebar.success(f"✅ Loaded {uploaded_file.name}")

    # Split text
    st.write("🔄 Splitting text into chunks...")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    st.write("🧠 Creating embeddings with Hugging Face model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma DB in memory
    vectordb = Chroma.from_documents(docs, embedding=embeddings)

    # Create retriever and LLM
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Chat input
    st.write("💬 Ask something about your document:")
    user_query = st.text_input("Your question:")

    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
        st.success("✅ Response:")
        st.write(response)

else:
    st.info("👆 Upload a file from the sidebar to begin.")
