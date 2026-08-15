import os
import dotenv
import google.generativeai as genai

dotenv.load_dotenv()

def get_gemini_response(context, query):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key.strip() in ("", "your_api_key_here", "your_actual_api_key_here"):
        return f"This is a fallback response for: '{query}'. Please set your Gemini API key in Streamlit Secrets or .env file to enable full AI responses."
    
    try:
        genai.configure(api_key=api_key)
        
        # Try latest model family with fallback
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Using the context given below, answer the query accurately and thoroughly.
                                        
        CONTEXT:
        {context}

        QUERY:
        {query}   

        Provide a clear, informative, and structured response based on the context.                             
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        # If gemini-1.5-flash fails, fallback to gemini-pro or gemini-1.5-pro
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"Error generating response: {str(e)}. Please check your API key."