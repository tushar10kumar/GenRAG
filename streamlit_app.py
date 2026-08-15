import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project services
from services.retrieve import retrieve_relevant_resources
from llm.get_response import ask

# Page configuration
st.set_page_config(
    page_title="GenRAG - AI-Powered Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    /* Main theme styling */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stApp {
        background-color: #0d1117;
    }
    
    /* Header banner */
    .header-container {
        background: linear-gradient(135deg, #161b22 0%, #1f242c 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #58a6ff;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .header-author {
        color: #6e7681;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Result card styling */
    .answer-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .chunk-box {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .score-badge {
        background-color: #238636;
        color: #ffffff;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def simple_sentence_split(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def auto_generate_embeddings(save_path):
    pdf_path = "data/The_intelligent_investor.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF document not found at {pdf_path}")
    
    from services.pdf_to_text import open_and_read_pdf
    from services.text_to_chucks import split_list
    from services.process_chunks import process_chunks
    from services.embed_model import embed_text

    pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
    for item in pages_and_texts:
        item["sentences"] = simple_sentence_split(item["text"])
        item["page_sentence_count_spacy"] = len(item["sentences"])
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=10)
        item["num_chunks"] = len(item["sentence_chunks"])

    pages_and_chunks = process_chunks(pages_and_texts=pages_and_texts)
    df = pd.DataFrame(pages_and_chunks)
    valid_chunks = df[df["chunk_token_count"] > 30].to_dict(orient="records")
    embed_text(valid_chunks)
    
    out_df = pd.DataFrame(valid_chunks)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out_df.to_csv(save_path, index=False)
    return out_df

# Load saved embeddings (cached for performance)
@st.cache_resource
def load_rag_data():
    embeddings_df_save_path = "data/text_chunks_and_embeddings_df.csv"
    if not os.path.exists(embeddings_df_save_path):
        df = auto_generate_embeddings(embeddings_df_save_path)
    else:
        df = pd.read_csv(embeddings_df_save_path)
        
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    pages_and_chunks = df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)
    return pages_and_chunks, embeddings, embeddings_df_save_path

# Header Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">⚡ GenRAG</div>
    <div class="header-subtitle">AI-Powered Document Intelligence & Retrieval System</div>
    <div class="header-author">Developed & Maintained by Tushar Kumar (Project by Ashish Prasad)</div>
</div>
""", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    gemini_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            gemini_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    
    if not gemini_key:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
    
    user_api_key = st.text_input(
        "Gemini API Key",
        value=gemini_key,
        type="password",
        help="Required to generate AI responses. If deployed on Streamlit Cloud, set this in App Secrets."
    )
    
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        st.success("API Key set successfully!", icon="✅")
    else:
        st.warning("Please provide a Gemini API Key to enable AI responses.", icon="⚠️")

    st.markdown("---")
    st.subheader("📚 Loaded Document")
    st.info("📄 **The Intelligent Investor** by Benjamin Graham")
    
    st.markdown("---")
    st.markdown("🔗 [GitHub Repository](https://github.com/tushar10kumar/GenRAG)")

# Main Query Interface
try:
    with st.spinner("Loading document embeddings..."):
        pages_and_chunks, embeddings, embeddings_df_save_path = load_rag_data()
    st.toast("Document embeddings loaded successfully!", icon="🚀")
except Exception as e:
    st.error(f"Failed to load document embeddings: {str(e)}")
    st.stop()

st.subheader("💬 Ask a Question")
query = st.text_input(
    "Enter your query about the document:",
    placeholder="e.g. What is margin of safety in investing?",
    key="query_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    search_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)

if search_button and query:
    with st.spinner("Searching document & generating AI answer..."):
        try:
            # 1. Retrieve top chunks & score
            scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)
            
            # 2. Generate response
            ans = ask(
                query=query,
                embeddings=embeddings,
                pages_and_chunks=pages_and_chunks,
                embeddings_df_save_path=embeddings_df_save_path
            )
            
            # Display Answer
            st.markdown("### 🤖 AI Response")
            st.markdown(f'<div class="answer-card">{ans}</div>', unsafe_allow_html=True)
            
            # Display Retrieved Context
            st.markdown("### 📖 Retrieved Document Context")
            for i, idx in enumerate(indices[:5]):
                item = pages_and_chunks[idx]
                score = float(scores[i].cpu().numpy())
                page_num = item.get("page_number", "N/A")
                text = item.get("sentence_chunk", "")
                
                with st.expander(f"Chunk {i+1} (Page {page_num}) — Similarity Score: {score:.4f}"):
                    st.write(text)
                    
        except Exception as e:
            st.error(f"An error occurred while processing your request: {str(e)}")
elif search_button and not query:
    st.warning("Please enter a question first!")
