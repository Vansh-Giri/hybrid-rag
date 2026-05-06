import os
import fitz  # Standard PyMuPDF
from typing import List, Dict

def load_pdf(file_path: str) -> List[Dict]:
    """Strictly extracts text using native PyMuPDF to bypass ONNX bugs."""
    documents = []
    print(f"Extracting from: {os.path.basename(file_path)}...")
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Use native markdown extraction if available, fallback to sorted text
            try:
                # PyMuPDF 1.24+ supports native markdown
                text = page.get_text("markdown")
            except Exception:
                text = page.get_text("text", sort=True)
                
            if text and text.strip():
                documents.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": file_path, 
                        "page": page_num + 1
                    }
                })
    except Exception as e:
        print(f"CRITICAL: Failed to load {os.path.basename(file_path)}: {e}")
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
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    print(f"Loading documents from: {os.path.abspath(test_dir)}")
    docs = load_directory(test_dir)
    print(f"\nSuccessfully loaded {len(docs)} document chunk(s)/page(s).")