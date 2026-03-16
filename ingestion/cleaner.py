import re
import unicodedata
from typing import List, Dict

def clean_text(text: str) -> str:
    """
    Advanced cleaning for technical PDF extractions.
    Removes artifacts, fixes hyphenation, and normalizes text[cite: 31, 32].
    """
    if not text:
        return ""
    
    # 1. Normalize Unicode (fixes weird PDF ligatures and smart quotes)
    text = unicodedata.normalize('NFKD', text)
    
    # 2. Fix hyphenated words split across lines (e.g., "informa-\ntion" -> "information")
    # This is critical for dense retrieval so the semantic meaning of the word isn't lost.
    text = re.sub(r'([a-zA-Z]+)-\n([a-zA-Z]+)', r'\1\2', text)
    
    # 3. Remove URLs (They create noise for embeddings and consume token limits)
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # 4. Remove standalone page numbers or single characters on a line (Header/Footer artifacts)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 5. Remove CID font artifacts (Often appear in technical PDFs as (cid:XX))
    text = re.sub(r'\(cid:\d+\)', '', text)

    # 6. Replace multiple newlines with a single newline to preserve paragraph boundaries
    text = re.sub(r'\n{2,}', '\n', text)
    
    # 7. Replace multiple spaces or tabs with a single space [cite: 31]
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 8. Strip non-ASCII/control characters (Keeps standard English punctuation/text)
    text = text.encode('ascii', 'ignore').decode('utf-8')

    return text.strip()

def clean_documents(documents: List[Dict]) -> List[Dict]:
    """Applies advanced cleaning to a list of document dictionaries."""
    cleaned_docs = []
    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        # Increased threshold: Drop chunks that are too short to contain semantic meaning
        if len(cleaned_text) > 50: 
            cleaned_docs.append({
                "text": cleaned_text,
                "metadata": doc["metadata"]
            })
    return cleaned_docs

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.loader import load_directory

    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_docs = load_directory(test_dir)
    
    cleaned_docs = clean_documents(raw_docs)
    print(f"Cleaned down to {len(cleaned_docs)} valid pages/chunks.")
    
    if cleaned_docs:
        print("\n--- Optimized Cleaned Output (First 1000 characters) ---")
        print(cleaned_docs[0]["text"][:1000])