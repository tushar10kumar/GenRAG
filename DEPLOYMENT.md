# GenRAG Deployment Guide

This document explains how to deploy **GenRAG** across various platforms.

---

## 1. Deploy on Streamlit Community Cloud (Recommended & Free)

Streamlit Community Cloud provides instant, free hosting directly from your GitHub repository.

### Step-by-Step Instructions:

1. **Push your code to GitHub**:
   Ensure your repository [`tushar10kumar/GenRAG`](https://github.com/tushar10kumar/GenRAG) has all updated files (`streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`).

2. **Go to Streamlit Community Cloud**:
   Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

3. **Create New App**:
   - Click **"New App"**.
   - Select your repository: `tushar10kumar/GenRAG`
   - Select branch: `main`
   - Set Main file path: `streamlit_app.py`

4. **Configure Secrets (Gemini API Key)**:
   In the **Secrets** modal, paste the following **TOML formatted** text (do not paste just the raw API key):

   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```

   > [!IMPORTANT]
   > Streamlit Secrets require standard **TOML key-value format**:
   > `GEMINI_API_KEY = "AQ.Ab8RN6..."`

5. **Deploy**:
   Click **Deploy!** Your app will be live on a public `.streamlit.app` URL in 1-2 minutes.

---

## 2. Deploy using Docker

You can run GenRAG anywhere Docker is supported (local, Render, Railway, AWS EC2, GCP Cloud Run, Azure Containers).

### Build & Run Locally with Docker:

```bash
# Build Docker image
docker build -t genrag .

# Run container with Gemini API Key
docker run -d -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" --name genrag_app genrag
```

Access the web app at: `http://localhost:8501`

---

## 3. Deploy on Render (1-Click Web Service)

1. Connect your GitHub repository to [Render.com](https://render.com/).
2. Create a new **Web Service** and select `tushar10kumar/GenRAG`.
3. Set Environment to **Python**.
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
6. Add Environment Variable:
   - Name: `GEMINI_API_KEY`
   - Value: `your_gemini_api_key`

---

## 4. Deploy on Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Push your repository code to the Space repository.
4. Add `GEMINI_API_KEY` under **Settings -> Repository secrets**.
