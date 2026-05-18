from .base_strategy import QueryStrategy
from .normal_v1 import NormalV1Strategy
from .paired_retrieval import PairedRetrievalStrategy

# Mapping ID/Tên hiển thị với Lớp chiến thuật tương ứng
STRATEGIES = {
    "Normal_v1 (Raw Query)": NormalV1Strategy,
    "Paired Retrieval (So sánh đôi)": PairedRetrievalStrategy,
}
