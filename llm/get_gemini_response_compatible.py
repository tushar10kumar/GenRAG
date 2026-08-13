import os
import dotenv
import google.ai.generativelanguage as glm
import google.generativeai as genai

dotenv.load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

if not gemini_api_key or gemini_api_key == "your_api_key_here":
    print("Warning: Gemini API key not set. Please add your API key to the .env file.")
    print("For now, returning a placeholder response.")

def get_gemini_response(context, query):
    if not gemini_api_key or gemini_api_key == "your_api_key_here":
        return f"This is a placeholder response for the query: '{query}'. Please set up your Gemini API key in the .env file to get actual AI responses."
    
    try:
        # Configure the API key
        genai.configure(api_key=gemini_api_key)
        
        # Create the model
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Using the context given below answer the query.
                                        
        CONTEXT: {context}

        QUERY: {query}   

        Make the answers long and informative.                             
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}. Please check your API key and try again."