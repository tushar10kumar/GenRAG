def get_simple_response(context, query):
    """
    Dynamic response fallback generator when LLM API is unavailable.
    Formats retrieved document evidence cleanly for any uploaded document.
    """
    response = f"""### 📖 Evidence Summary for Query: *"{query}"*

**Retrieved Knowledge Base Context:**

{context}

---
*Note: The evidence above was retrieved directly from your active document knowledge base. For enhanced generative AI responses, ensure your Gemini API Key is configured.*
"""
    return response