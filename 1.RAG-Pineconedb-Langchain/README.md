# 🧠 Hybrid Search with Pinecone, BM25, and HuggingFace Embeddings (LangChain)

This project demonstrates a **hybrid search system** using:

- **Dense vector embeddings** via `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
- **Sparse retrieval** using **BM25** via `pinecone-text`
- **Hybrid indexing and querying** using **Pinecone**
- Integration with **LangChain's PineconeHybridSearchRetriever**

---

## 🚀 Features

- Combines semantic search (dense vectors) and keyword search (BM25) for improved relevance.
- Uses `LangChain` for unified retriever interface.
- Indexing and querying in **Pinecone** for scalable vector similarity search.
- BM25 model trained on your own text corpus and saved/loaded via JSON.

---

## 📦 Dependencies

Make sure to install the following packages:

```bash
pip install python-dotenv pinecone-client langchain langchain-community pinecone-text langchain-huggingface
```
## 🛠️ Environment Variables

Create a .env file in the root directory:

```bash
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
```

## 📌 Notes

- Make sure Pinecone index region matches your Pinecone project region.
- Ensure HuggingFace token allows model access (all-MiniLM-L6-v2 is public).
- BM25 encoder must be trained on the same texts you index.

## 🤝 Credits

- [Pinecone](https://www.pinecone.io/)
- [LangChain](https://www.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/models)
- [pinecone-text GitHub](https://github.com/pinecone-io/pinecone-text)
