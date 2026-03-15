# Khởi tạo package
from .document_processor import process_document
from .legal_chunker import LegalChunker, DocumentChunk

__all__ = ["process_document", "LegalChunker", "DocumentChunk"]
