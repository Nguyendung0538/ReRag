import pdfplumber

def load_pdf(file_path: str) -> str:
    """
    Reads a PDF file and extracts text using pdfplumber, keeping paragraph structures as much as possible.
    Returns the full text as a string.
    """
    full_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Extract text preserving layout to some extent
                text = page.extract_text(layout=True)
                if text:
                    # Basic cleanup of multiple empty lines
                    lines = [line.strip() for line in text.split('\n')]
                    clean_lines = [line for line in lines if line]
                    full_text.extend(clean_lines)
                    
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return ""

if __name__ == "__main__":
    pass
