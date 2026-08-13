#!/usr/bin/env python3
"""
Simple test script for GenRAG system
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from services.retrieve import print_wrapped, print_top_results_and_scores
from llm.get_response import ask

def test_rag_system():
    """Test the RAG system with predefined queries"""
    
    device = "cpu"
    print("Loading the Saved Embeddings DataFrame...")
    
    embeddings_df_save_path = "data/text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df = pd.read_csv(embeddings_df_save_path)
    
    print("Converting the 'embedding' column to a numpy array...")
    text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    
    pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to(device)
    
    print("Successfully loaded embeddings!")
    print("\n" + "="*50)
    
    # Test queries
    test_queries = [
        "What is value investing?",
        "What are the principles of investing?",
        "How to analyze stocks?",
        "What is margin of safety?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Show top results
        print_top_results_and_scores(query=query, embeddings=embeddings, pages_and_chunks=pages_and_chunks, n_resources_to_return=2)
        
        # Get AI response
        print("\nAI Response:")
        ans = ask(query=query, embeddings=embeddings, pages_and_chunks=pages_and_chunks, embeddings_df_save_path=embeddings_df_save_path)
        print_wrapped(ans)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_rag_system()