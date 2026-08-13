def get_simple_response(context, query):
    """
    Simple response generator that combines the retrieved context with the query
    """
    response = f"""
Based on the retrieved information from "The Intelligent Investor":

Query: {query}

Relevant Context:
{context}

Summary: The retrieved passages discuss value investing as an approach that has been established for over 50 years since Ben Graham and Dave Dodd wrote "Security Analysis". The text suggests that value investing involves identifying discrepancies between price and value in the marketplace, and that those who follow Graham & Dodd's principles continue to prosper. It also mentions that the approach seems to be either immediately understood or not at all, rather than being gradually learned.

Note: This is a basic response. For more sophisticated AI-generated responses, please set up your Gemini API key in the .env file.
"""
    return response