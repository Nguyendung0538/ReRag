import pdfplumber

def load_pdf(file_source) -> str:
    """
    Reads a PDF file from a path string or a file-like stream object using pdfplumber.
    Returns the full text as a string.
    """
    full_text = []
    try:
        with pdfplumber.open(file_source) as pdf:
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
        print(f"Error loading PDF: {e}")
        return ""

if __name__ == "__main__":
    pass
