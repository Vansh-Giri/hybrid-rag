import os
import requests
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()

class RAGGenerator:
    # Updated default models based on current API availability
    def __init__(self, gemini_model="gemini-2.5-flash", ollama_model="phi4-mini:latest", groq_model="llama-3.1-8b-instant"):
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        self.groq_model = groq_model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Hardened system prompt to force exact mathematical output and strip conversational fluff
        self.system_instruction = (
            "You are a strict, precise technical assistant. Use ONLY the provided context to answer. "
            "CRITICAL: If the user asks for a formula, equation, or code, extract it exactly as it appears in the context. "
            "Wrap all mathematical formulas in standard LaTeX delimiters ($$ equation $$). "
            "Do not add conversational filler. If the answer is not explicitly in the context, output exactly: 'Information not found in context.'"
        )
        
        # Initialize Clients
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
        
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

    def _build_prompt(self, query: str, context_chunks: List[Tuple[Dict, float]], history: List[Dict] = None) -> str:
        context_text = "\n\n---\n\n".join(
            [f"Source: {chunk['metadata'].get('source', 'Unknown')} (Page {chunk['metadata'].get('page', 'N/A')})\n{chunk['text']}" 
             for chunk, _ in context_chunks]
        )
        
        history_text = ""
        if history:
            recent_history = history[-3:] 
            history_text = "Previous Conversation:\n" + "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in recent_history]
            ) + "\n\n"
        
        return f"{history_text}Context:\n{context_text}\n\nCurrent Question: {query}\n\nAnswer:"

    def _generate_gemini(self, prompt: str) -> str:
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured.")
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0.0, # Zero temperature for strict extraction
            )
        )
        return response.text

    def _generate_groq(self, prompt: str) -> str:
        if not self.groq_client:
            raise ValueError("Groq API key not configured.")
        
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": prompt}
            ],
            model=self.groq_model,
            temperature=0.0,
        )
        return chat_completion.choices[0].message.content

    def _generate_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": f"{self.system_instruction}\n\n{prompt}",
            "stream": False,
            "options": {"num_ctx": 2048, "temperature": 0.0, "stop": ["\nQuestion:", "Current Question:"]}
        }
        response = requests.post(self.ollama_url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json().get("response", "")

    def generate_answer(self, query: str, context_chunks: List[Tuple[Dict, float]], history: List[Dict] = None, provider: str = "gemini") -> Tuple[str, bool]:
        if not context_chunks:
            return "No relevant context found to answer the query.", False

        prompt = self._build_prompt(query, context_chunks, history)
        used_fallback = False
        
        try:
            if provider == "gemini":
                return self._generate_gemini(prompt), used_fallback
            elif provider == "groq":
                return self._generate_groq(prompt), used_fallback
            else:
                return self._generate_ollama(prompt), used_fallback
                
        except Exception as primary_e:
            print(f"Primary provider '{provider}' failed: {primary_e}. Triggering fallback to Groq...")
            used_fallback = True
            try:
                # Fallback to Groq (fastest, most reliable)
                return self._generate_groq(prompt), used_fallback
            except Exception as fallback_e:
                return f"System Failure. Primary Error: {primary_e} | Fallback Error: {fallback_e}", used_fallback