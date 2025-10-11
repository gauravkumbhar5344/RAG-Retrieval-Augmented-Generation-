# 🤖 RAG Chatbot using LangChain, Groq & Hugging Face Embeddings

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Groq**, and **Hugging Face** sentence-transformer embeddings.  
It allows users to **upload a file (PDF, TXT, DOCX)** and **chat** with its contents through a simple **Streamlit** interface.

---

## 🚀 Features
- 🧠 Uses **Hugging Face embeddings** 
- ⚙️ Powered by **Groq LLaMA3** for natural responses
- 📄 Supports **PDF, TXT, and DOCX** uploads
- 🔍 Retrieves relevant chunks using **Chroma vector store**
- 💬 Interactive chat interface built with **Streamlit**

---

## 🧩 Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/gauravkumbhar5344/RAG-Retrieval-Augmented-Generation-.git
   cd 2..RAG_Using_HuggingFace
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up your Groq API Key**
   - Get your key from https://console.groq.com/keys
   - Create a .env file in your project folder:
   ```bash
   GROQ_API_KEY=your_api_key_here
   ```
4. **Run the App**
   ```bash
   streamlit run app.py
   ```
