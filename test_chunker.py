from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.document_parser import DocumentParser

parser = DocumentParser()
chunker = LegalChunker()

file2 = r"d:\Code\ReRag\document\test\HỢP ĐỒNG THUÊ NHÀ No10_LK52_08.04.2026.docx"

text2 = parser.parse(file2)
chunks2 = chunker.chunk(text2, source="Bản Mới")
print(f"Bản Mới has {len(chunks2)} chunks")
found = False
for c in chunks2:
    if "6" in c.metadata.get("dieu", ""):
        print(f"Found something with 6 in Bản Mới: {c.metadata['dieu']}")
        found = True

if not found:
    print("NO Điều 6 was found in metadata for Bản Mới. Dumping all Dieu metadatas:")
    dieus = set(c.metadata.get("dieu", "None") for c in chunks2)
    print(dieus)
