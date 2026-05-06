import os
import google.generativeai as genai
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

class RAGGenerator:
    # CHANGED: Use -latest to avoid 404 routing errors
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction="You are a helpful and precise technical assistant. Use ONLY the provided retrieved context to answer the user's question. If the answer is not contained within the context, say 'I don't have enough information in the provided documents to answer that.' Do not hallucinate."
        )
        
    def _build_prompt(self, query: str, context_chunks: List[Tuple[Dict, float]], history: List[Dict] = None) -> str:
        """Constructs the payload string using retrieved context and previous chat history."""
        context_text = "\n\n---\n\n".join(
            [f"Source: {chunk['metadata'].get('source', 'Unknown')} (Page {chunk['metadata'].get('page', 'N/A')})\n{chunk['text']}" 
             for chunk, _ in context_chunks]
        )
        
        # Inject the last 3 conversational turns to maintain context
        history_text = ""
        if history:
            recent_history = history[-3:] 
            history_text = "Previous Conversation:\n" + "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in recent_history]
            ) + "\n\n"
        
        prompt = f"""{history_text}Context:
{context_text}

Current Question: {query}

Answer:"""
        return prompt

    def generate_answer(self, query: str, context_chunks: List[Tuple[Dict, float]], history: List[Dict] = None) -> str:
        """Sends the prompt and history to the Gemini API."""
        if not context_chunks:
            return "No relevant context found to answer the query."

        prompt = self._build_prompt(query, context_chunks, history)
        
        print(f"Sending prompt to cloud API ({self.model_name})...")
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            return response.text
        except Exception as e:
            return f"Error during generation: {str(e)}"