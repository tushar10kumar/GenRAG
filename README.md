# GenRAG

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

GenRAG is an AI-powered document intelligence system that implements Retrieval-Augmented Generation (RAG) from scratch. Built without high-level frameworks like LangChain or vector databases, it features a modern GUI, semantic search through 1,400+ text chunks, and AI-powered responses. The system processes PDFs, generates embeddings, and provides instant answers to questions about document content with sub-millisecond search speeds.

## Features

- 🎨 **Modern GUI**: GitHub-inspired dark theme interface
- 📄 **PDF Processing**: Automatic text extraction and chunking
- 🧠 **Smart Embeddings**: Sentence-BERT powered semantic search
- ⚡ **Fast Search**: Sub-millisecond query processing
- 🤖 **AI Responses**: Google Gemini API integration
- 💾 **Lightweight**: CSV-based storage (no vector database needed)
- 🔧 **Python 3.8+ Compatible**: Works on older Python versions


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AkprasadoP/GenRAG.git
    cd GenRAG
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    call env\Scripts\activate  # On Windows
    # source env/bin/activate  # On Linux/Mac
    ```

3. Install compatible dependencies:
    ```bash
    pip install -r requirements_minimal.txt
    ```

## Setup

1. **Generate Embeddings** (The PDF is already included):
    ```bash
    python create_embeddings_auto.py
    ```
   This will process "The Intelligent Investor" PDF and create embeddings.

2. **Configure Gemini API** (Optional for enhanced responses):
   - Edit `.env` file and add your API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

## Usage

### GUI Application (Recommended)
```bash
python app.py
```
Or simply double-click `run.bat`

### Terminal Version
```bash
python main.py
```

### Test Multiple Queries
```bash
python test_rag.py
```




## LLM Response

You can use both a local LLM or an LLM from an API like Gemini for generating responses.

- **Local LLM**: If you have the capability to run a local LLM, you can use it for generating responses. Cause mine is too slow :(

- **LLM from API**: If your system is not powerful enough for local inference, you can use an API like Gemini. To do this, create a `.env` file and pass the Gemini API key.

### Using Gemini API

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Edit the existing `.env` file:
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```
3. Restart the application to use enhanced AI responses

**Note**: The system works perfectly with fallback responses even without the API key.

---



## Author

**Ashish Prasad** - [@AkprasadoP](https://github.com/AkprasadoP)

This project is developed and maintained by Ashish Prasad.



## Credits

Special thanks to the following YouTube channels and research papers for their invaluable resources and insights:

### YouTube Channels

- [AI Anytime](https://www.youtube.com/@AIAnytime)
- [Daniel Brouke](https://www.youtube.com/@mrdbourke)
- [Krish Naik](https://www.youtube.com/@krishnaik06)
- [Codebasics](https://www.youtube.com/@codebasics)
- [CampusX](https://www.youtube.com/@campusx-official)

### Research Papers

- Patrick Lewis ., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Vaswani et al., "Attention is All You Need" [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Reimers et al., "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
