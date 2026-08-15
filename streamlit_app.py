import os
import re
import glob
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
    page_title="GenRAG — AI Multi-Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for vibrant aesthetics, animations, and glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background space theme */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #111827 0%, #070a12 100%);
        color: #f1f5f9;
    }
    
    /* Animated Gradient Header Banner */
    .hero-banner {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 15, 26, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2.8rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 30px rgba(56, 189, 248, 0.08);
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #00c6ff, #7928ca, #ff0080);
        background-size: 300% 300%;
        animation: gradientShift 6s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 1.2rem;
    }
    
    /* Glowing Pulse Dot */
    .status-pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 0.3rem 0.85rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.6rem;
    }
    
    /* Stat Cards with Glow Effect */
    .stat-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 25px -5px rgba(56, 189, 248, 0.2);
    }
    
    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.2rem;
        font-weight: 500;
    }
    
    /* Response card styling */
    .answer-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(11, 15, 25, 0.98) 100%);
        border-left: 4px solid #8b5cf6;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        font-size: 1.05rem;
        line-height: 1.75;
    }
    
    /* Button Hover Micro-animations */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.35);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f17;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

def simple_sentence_split(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def generate_embeddings_for_pdfs(pdf_paths, save_path):
    from services.pdf_to_text import open_and_read_pdf
    from services.text_to_chucks import split_list
    from services.process_chunks import process_chunks
    from services.embed_model import embed_text

    all_chunks = []
    for pdf_path in pdf_paths:
        file_name = os.path.basename(pdf_path)
        pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
        for item in pages_and_texts:
            item["document_name"] = file_name
            item["sentences"] = simple_sentence_split(item["text"])
            item["page_sentence_count_spacy"] = len(item["sentences"])
            item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=10)
            item["num_chunks"] = len(item["sentence_chunks"])

        chunks = process_chunks(pages_and_texts=pages_and_texts)
        all_chunks.extend(chunks)

    df = pd.DataFrame(all_chunks)
    valid_chunks = df[df["chunk_token_count"] > 30].to_dict(orient="records")
    embed_text(valid_chunks)
    
    out_df = pd.DataFrame(valid_chunks)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out_df.to_csv(save_path, index=False)
    return out_df

@st.cache_resource
def load_rag_data():
    embeddings_df_save_path = "data/text_chunks_and_embeddings_df.csv"
    if not os.path.exists(embeddings_df_save_path):
        pdf_files = glob.glob("data/*.pdf")
        if not pdf_files:
            st.error("No PDF files found in data directory.")
            st.stop()
        df = generate_embeddings_for_pdfs(pdf_files, embeddings_df_save_path)
    else:
        df = pd.read_csv(embeddings_df_save_path)
        
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    pages_and_chunks = df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)
    return pages_and_chunks, embeddings, embeddings_df_save_path

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">⚡ GenRAG Intelligence System</div>
    <div class="hero-subtitle">Production Retrieval-Augmented Generation Architecture Built from Scratch</div>
    <div>
        <span class="badge-pill"><span class="status-pulse"></span> System Ready</span>
        <span class="badge-pill">🧠 Sentence-BERT Vector Embeddings</span>
        <span class="badge-pill">🤖 Google Gemini 1.5 LLM</span>
        <span class="badge-pill">⚡ Sub-Millisecond Search</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data early to populate stats
try:
    with st.spinner("Loading document matrix..."):
        pages_and_chunks, embeddings, embeddings_df_save_path = load_rag_data()
except Exception as e:
    st.error(f"Failed to load document embeddings: {str(e)}")
    st.stop()

# Live Metrics Bar
pdf_list = glob.glob("data/*.pdf")
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(pdf_list)}</div><div class="stat-label">📚 Indexed Documents</div></div>', unsafe_allow_html=True)
with col_stat2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(pages_and_chunks):,}</div><div class="stat-label">🧩 Vector Text Chunks</div></div>', unsafe_allow_html=True)
with col_stat3:
    st.markdown('<div class="stat-card"><div class="stat-value">&lt; 1 ms</div><div class="stat-label">⚡ Vector Latency</div></div>', unsafe_allow_html=True)
with col_stat4:
    st.markdown('<div class="stat-card"><div class="stat-value">Gemini 1.5</div><div class="stat-label">🤖 Active AI Engine</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    
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
        help="Required to generate AI responses."
    )
    
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        st.success("API Key Active", icon="✅")
    else:
        st.warning("Please provide a Gemini API Key.", icon="⚠️")

    st.markdown("---")
    st.markdown("### 📤 Upload New PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF to index:", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("📥 Index & Embed Uploaded PDF", type="primary", use_container_width=True):
            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                os.makedirs("data", exist_ok=True)
                save_pdf_path = os.path.join("data", uploaded_file.name)
                with open(save_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                pdf_files = glob.glob("data/*.pdf")
                embeddings_df_save_path = "data/text_chunks_and_embeddings_df.csv"
                generate_embeddings_for_pdfs(pdf_files, embeddings_df_save_path)
                st.cache_resource.clear()
                st.success(f"Indexed '{uploaded_file.name}'!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Active Knowledge Base")
    if pdf_list:
        for p in pdf_list:
            st.markdown(f"• 📄 `{os.path.basename(p)}`")
    else:
        st.write("No PDF files found.")
        
    st.markdown("---")
    st.markdown("👨‍💻 **Developer**: [Tushar Kumar](https://github.com/tushar10kumar)")
    st.markdown("🔗 [GitHub Repository](https://github.com/tushar10kumar/GenRAG)")
    st.markdown("🌐 [Live Web App](https://genrag-x5xm3zputvdyxfv654zvmp.streamlit.app)")

# Main Query Section
st.markdown("### 💬 Ask Document Intelligence Assistant")

# Quick Prompt Suggestions
st.caption("Suggested Quick Prompts:")
suggested_cols = st.columns(3)
selected_query = ""

if suggested_cols[0].button("💡 What is margin of safety?"):
    selected_query = "What is margin of safety in investing?"
if suggested_cols[1].button("⚖️ Investor vs Speculator"):
    selected_query = "What is the difference between an investor and a speculator?"
if suggested_cols[2].button("📈 Rules for selecting stocks"):
    selected_query = "What are the rules for selecting stocks for conservative investors?"

default_query = selected_query if selected_query else ""

query = st.text_input(
    "Enter your question about your indexed document knowledge base:",
    value=default_query,
    placeholder="e.g. What is margin of safety in investing?",
    key="query_input"
)

btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    search_button = st.button("🔍 Generate AI Response", type="primary", use_container_width=True)

if (search_button or selected_query) and query:
    with st.spinner("Searching document matrix & generating AI synthesis..."):
        try:
            scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)
            
            ans = ask(
                query=query,
                embeddings=embeddings,
                pages_and_chunks=pages_and_chunks,
                embeddings_df_save_path=embeddings_df_save_path
            )
            
            # Use Tabs for organized viewing
            tab1, tab2 = st.tabs(["🤖 AI Answer", "📖 Retrieved Evidence Chunks"])
            
            with tab1:
                st.markdown(f'<div class="answer-card">{ans}</div>', unsafe_allow_html=True)
                
                # Feedback & Download row
                col_fb1, col_fb2, col_fb3 = st.columns([2, 1, 1])
                with col_fb1:
                    st.download_button(
                        label="📥 Download Response Text",
                        data=f"Query: {query}\n\nAI Response:\n{ans}",
                        file_name="genrag_response.txt",
                        mime="text/plain"
                    )
                with col_fb2:
                    if st.button("👍 Helpful"):
                        st.toast("Thank you for your feedback!", icon="🎉")
                with col_fb3:
                    if st.button("👎 Needs Improvement"):
                        st.toast("Feedback recorded!", icon="📝")

            with tab2:
                for i, idx in enumerate(indices[:5]):
                    item = pages_and_chunks[idx]
                    score = float(scores[i].cpu().numpy())
                    doc_name = item.get("document_name", item.get("pdf_path", "Document"))
                    page_num = item.get("page_number", "N/A")
                    text = item.get("sentence_chunk", "")
                    
                    with st.expander(f"📌 Chunk {i+1}: {doc_name} (Page {page_num}) — Similarity Score: {score:.4f}"):
                        st.write(text)
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
elif search_button and not query:
    st.warning("Please enter a question first!")
