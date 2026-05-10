import os
import pdfplumber

def load_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [{
            "text": f.read(), 
            "metadata": {
                "source": os.path.basename(file_path), 
                "page": 1, 
                "type": "txt"
            }
        }]

def load_pdf(file_path: str) -> list[dict]:
    pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages.append({
                        "text": text,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "page": i + 1,
                            "type": "pdf"
                        }
                    })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return pages

def load_directory(data_dir: str = "data") -> list[dict]:
    all_documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.lower().endswith(".txt"):
            all_documents.extend(load_txt(file_path))
        elif filename.lower().endswith(".pdf"):
            all_documents.extend(load_pdf(file_path))
    return all_documents