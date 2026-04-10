from .base_indexing import BaseIndexingStrategy
from .tradi_rag import TradiRAGIndexing
from .hierarchical_rag_vectorless import HierarchicalRAGIndexing as VectorlessRAG
from .hierarchical_rag import HierarchicalRAGIndexing as HybridRAG

INDEXING_STRATEGIES = {
    "TradiRAG (Chunk + Embed + VectorDB)": TradiRAGIndexing,
    "Hierarchical (Vectorless PageIndex)": VectorlessRAG,
    "Hierarchical RAG (Hybrid Vector & Summaries)": HybridRAG,
}
