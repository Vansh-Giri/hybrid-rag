import os
import fitz  # PyMuPDF
from typing import List, Dict

def load_pdf(file_path: str) -> List[Dict]:
    """Extracts text from a PDF file page by page."""
    documents = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {"source": file_path, "page": page_num + 1}
                })
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
    return documents

def load_txt(file_path: str) -> List[Dict]:
    """Extracts text from a TXT file."""
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {"source": file_path}
                })
    except Exception as e:
        print(f"Error loading TXT {file_path}: {e}")
    return documents

def load_directory(directory_path: str) -> List[Dict]:
    """Loads all supported documents from a directory."""
    all_documents = []
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return all_documents

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.lower().endswith('.pdf'):
            all_documents.extend(load_pdf(file_path))
        elif filename.lower().endswith('.txt'):
            all_documents.extend(load_txt(file_path))
    
    return all_documents

if __name__ == "__main__":
    # Test by printing text
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a temporary txt file for testing
    test_file = os.path.join(test_dir, "test_ingestion.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a test document to verify the ingestion pipeline works.\nIt supports multiple lines.")
        
    print(f"Loading documents from: {os.path.abspath(test_dir)}")
    docs = load_directory(test_dir)
    
    print(f"\nSuccessfully loaded {len(docs)} document chunk(s)/page(s).")
    if docs:
        print("\n--- Test Output ---")
        print(f"Source: {docs[0]['metadata']['source']}")
        print(f"Text Content: {docs[0]['text']}")