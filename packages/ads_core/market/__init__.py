"""Market targets for outcome validation.

Provides:
- Canonical market categories based on O*NET/ESCO taxonomy
- Market target vectors for similarity computation
- Category-to-major mappings
"""
from ads_core.market.targets import (
    MarketCategory,
    MARKET_CATEGORIES,
    get_market_category_texts,
    create_market_target_vectors,
    compute_market_similarity,
)

__all__ = [
    "MarketCategory",
    "MARKET_CATEGORIES",
    "get_market_category_texts",
    "create_market_target_vectors",
    "compute_market_similarity",
]
