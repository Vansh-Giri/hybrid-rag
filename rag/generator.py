import os
import sys
import requests
from typing import List, Dict, Tuple
from google import genai
from google.genai import types
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from utils.logger import setup_logger

logger = setup_logger("Generator")

class RAGGenerator:
    def __init__(self, 
                 gemini_model=settings.GEMINI_MODEL, 
                 ollama_model=settings.OLLAMA_MODEL, 
                 groq_model=settings.GROQ_MODEL):
        
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        self.groq_model = groq_model
        self.ollama_url = settings.OLLAMA_URL
        
        
        self.system_instruction = (
            "You are a precise technical assistant. Answer the user's question using ONLY the provided context. "
            "WARNING: The context is extracted from academic PDFs using OCR. It may contain noisy formatting, missing spaces, or fragmented math symbols (e.g., 'Q K T' instead of QK^T). "
            "Do your best to logically interpret these noisy fragments and reconstruct the intended mathematical equations. "
            "FORMATTING RULE: Format all reconstructed equations using standard LaTeX wrapped in double dollar signs (e.g., $$ E = mc^2 $$). "
            "CRITICAL INSTRUCTIONS: "
            "1. Answer directly, confidently, and concisely. "
            "2. DO NOT use phrases like 'Based on the context', 'According to the document', 'The provided text states', or 'I can infer'. "
            "3. Act as if you inherently know this information. "
            "4. If the concept is genuinely missing from the context, output exactly: 'I do not have enough information in the knowledge base to answer that.' Do not elaborate or apologize."
        )
        
        self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

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
            "options": {
                "num_ctx": 2048, 
                "temperature": 0.0, 
                "top_k": 40,       # Relaxed to allow natural text generation
                "top_p": 0.9,      # Relaxed to prevent repetitive loops
                "stop": ["\nQuestion:", "Current Question:"]
            }
        }
        
        response = requests.post(self.ollama_url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def generate_answer(self, query: str, context_chunks: List[Tuple[Dict, float]], history: List[Dict] = None, provider: str = "gemini") -> Tuple[str, bool]:
        if not context_chunks:
            logger.warning(f"No context chunks provided for query: '{query}'")
            return "No relevant context found to answer the query.", False

        prompt = self._build_prompt(query, context_chunks, history)
        used_fallback = False
        
        try:
            logger.info(f"Generating answer using primary provider: {provider.upper()}")
            if provider == "gemini":
                return self._generate_gemini(prompt), used_fallback
            elif provider == "groq":
                return self._generate_groq(prompt), used_fallback
            else:
                return self._generate_ollama(prompt), used_fallback
                
        except Exception as primary_e:
            logger.error(f"Primary provider '{provider}' failed: {primary_e}")
            logger.warning("Triggering fallback to Groq...")
            used_fallback = True
            try:
                # Fallback to Groq (fastest, most reliable)
                return self._generate_groq(prompt), used_fallback
            except Exception as fallback_e:
                logger.critical(f"System Failure! Primary Error: {primary_e} | Fallback Error: {fallback_e}")
                return f"System Failure. Primary Error: {primary_e} | Fallback Error: {fallback_e}", used_fallback