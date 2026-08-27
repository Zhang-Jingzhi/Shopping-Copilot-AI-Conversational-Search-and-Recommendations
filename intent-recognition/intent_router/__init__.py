"""Intent routing and query understanding for the TechJam shopping agent."""

from .catalog_lexicon import load_catalog_brands, load_catalog_categories
from .models import IntentResult
from .router import IntentRouter

__all__ = ["IntentResult", "IntentRouter", "load_catalog_brands", "load_catalog_categories"]
