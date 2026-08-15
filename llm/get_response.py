from llm.get_gemini_response_compatible import get_gemini_response
from llm.simple_response import get_simple_response
from services.retrieve import retrieve_relevant_resources

def ask(query, embeddings, pages_and_chunks, embeddings_df_save_path, n_resources_to_return=5):
    """
    Retrieves top relevant text chunks across all loaded PDF documents and passes structured evidence to Gemini.
    """
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        n_resources_to_return=min(n_resources_to_return, len(pages_and_chunks))
    )
    
    context_items = [pages_and_chunks[i] for i in indices]

    # Format context with explicit document source tags for multi-PDF intelligence
    formatted_chunks = []
    for item in context_items:
        doc_name = item.get("document_name", item.get("pdf_path", "Document"))
        page_num = item.get("page_number", "N/A")
        text = item.get("sentence_chunk", "")
        formatted_chunks.append(f"[Source: {doc_name} | Page: {page_num}]\n{text}")

    context = "\n\n".join(formatted_chunks)

    try:
        ans = get_gemini_response(query=query, context=context)
        if "Error generating response" in ans or "placeholder response" in ans:
            ans = get_simple_response(context=context, query=query)
    except Exception as e:
        ans = get_simple_response(context=context, query=query)

    return ans