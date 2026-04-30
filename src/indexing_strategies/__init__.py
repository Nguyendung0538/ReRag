from .base_indexing import BaseIndexingStrategy
from .tradi_rag import TradiRAGIndexing

INDEXING_STRATEGIES = {
    "Traditional RAG": TradiRAGIndexing,
}
