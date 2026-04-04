from .base_strategy import QueryStrategy
from .normal_v1 import NormalV1Strategy
from .decompose_v1 import DecomposeV1Strategy

# Mapping ID/Tên hiển thị với Lớp chiến thuật tương ứng
STRATEGIES = {
    "Normal_v1 (Raw Query)": NormalV1Strategy,
    "Decompose_v1 (Intent + Sub-query)": DecomposeV1Strategy,
}
