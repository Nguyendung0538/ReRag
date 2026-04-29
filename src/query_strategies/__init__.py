from .base_strategy import QueryStrategy
from .normal_v1 import NormalV1Strategy

STRATEGIES = {
    "Normal_v1 (Raw Query)": NormalV1Strategy,
}
