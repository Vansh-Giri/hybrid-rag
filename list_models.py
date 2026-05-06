import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit()

genai.configure(api_key=api_key)

print("Available Models supporting generateContent:\n" + "-"*45)

# Iterate through available models and print their names
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)