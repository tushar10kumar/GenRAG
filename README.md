# GenRAG ⚡

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genrag-x5xm3zputvdyxfv654zvmp.streamlit.app)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**GenRAG** is an AI-powered multi-document intelligence system implementing Retrieval-Augmented Generation (RAG) **from scratch**. Built without heavy abstractions like LangChain or complex vector database servers, it features a glassmorphic dark theme GUI, semantic search through 1,400+ vector text chunks, multi-PDF document indexing, and sub-millisecond query processing.

---

## 🌐 Live Web Application

🚀 **Experience GenRAG Live**: [**genrag-x5xm3zputvdyxfv654zvmp.streamlit.app**](https://genrag-x5xm3zputvdyxfv654zvmp.streamlit.app)

---

## ✨ Features

- 🎨 **Modern Cyberpunk UI**: Cyberpunk dark theme with glassmorphic cards, micro-animations, and live metric stats.
- 📄 **Multi-PDF Support**: Index multiple PDF documents automatically or upload new PDFs directly through the Web UI.
- 🧠 **Smart Embeddings**: Sentence-BERT (`all-mpnet-base-v2`) powered semantic vector search.
- ⚡ **Sub-Millisecond Speed**: Lightning-fast dot-product vector retrieval on text chunks.
- 🤖 **AI Responses**: Powered by Google Gemini 1.5 LLM with dynamic key loading & fallback model handling.
- 💾 **Lightweight Architecture**: Self-contained CSV vector storage without external vector database setup.
- 🐳 **Docker & Multi-Cloud Ready**: Complete `Dockerfile`, `render.yaml`, and `Procfile` included.

---

## 🚀 Quick Start (Local Setup)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tushar10kumar/GenRAG.git
   cd GenRAG
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv env
   call env\Scripts\activate  # On Windows
   # source env/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Gemini API Key** (in `.env` file):
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

5. **Run the Streamlit Web App**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🐳 Docker Deployment

To build and run GenRAG inside a lightweight Docker container:

```bash
# Build the Docker image
docker build -t genrag .

# Run container on port 8501
docker run -d -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" --name genrag_app genrag
```

Access the app at: `http://localhost:8501`

---

## 📖 Project Links & References

- 🌐 **Live Web Application**: [genrag-x5xm3zputvdyxfv654zvmp.streamlit.app](https://genrag-x5xm3zputvdyxfv654zvmp.streamlit.app)
- 🐙 **GitHub Repository**: [tushar10kumar/GenRAG](https://github.com/tushar10kumar/GenRAG)
- 📚 **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 👨‍💻 Author

Developed and maintained by **Tushar Kumar** — [@tushar10kumar](https://github.com/tushar10kumar)

---

## 📜 Credits & Acknowledgments

- **AI Anytime**: RAG concepts & architecture insights
- **Daniel Bourke**: PyTorch & Deep Learning workflows
- **Krish Naik & CampusX**: Generative AI & NLP tutorials
- **Research Papers**:
  - Lewis et al., *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* [(arXiv:2005.11401)](https://arxiv.org/abs/2005.11401)
  - Reimers et al., *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"* [(arXiv:1908.10084)](https://arxiv.org/abs/1908.10084)

---

## 📄 License

Licensed under the [MIT License](LICENSE).
