import docx
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

def iter_block_items(parent):
    """
    Yield each paragraph and table child within *parent*, in document order.
    Each returned value is an instance of either Table or Paragraph.
    """
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Something's not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def load_docx(file_path: str) -> str:
    """
    Reads a DOCX file and extracts text from paragraphs and tables in the exact document order.
    Returns the full text as a string.
    """
    try:
        doc = docx.Document(file_path)
        full_text = []
        
        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                if block.text.strip():
                    full_text.append(block.text.strip())
            elif isinstance(block, Table):
                for row in block.rows:
                    row_data = []
                    for cell in row.cells:
                        clean_cell = cell.text.strip().replace('\n', ' ')
                        if clean_cell and clean_cell not in row_data:
                            row_data.append(clean_cell)
                    if row_data:
                        full_text.append(" | ".join(row_data))
                        
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error loading DOCX {file_path}: {e}")
        return ""

if __name__ == "__main__":
    # Test simple run
    pass
