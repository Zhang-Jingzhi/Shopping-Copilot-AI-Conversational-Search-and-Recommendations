"""Catalog-derived vocabulary used by the deterministic conversation parser."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def message_ngrams(text: str, maximum_words: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())
    return {
        " ".join(words[start : start + width])
        for start in range(len(words))
        for width in range(1, min(maximum_words, len(words) - start) + 1)
    }


@dataclass(frozen=True)
class CatalogLexicon:
    """Normalized category, brand, and catalog-present feature vocabulary."""

    categories: dict[str, str]
    brands: dict[str, str]
    feature_terms: frozenset[str]

    @classmethod
    def from_jsonl(cls, catalog_path: str | Path) -> "CatalogLexicon":
        return _load_catalog_lexicon(str(Path(catalog_path).resolve()))

    def match_categories(self, text: str) -> list[str]:
        matches = self._matches(self.categories, text)
        return [self.categories[term] for term in matches]

    def match_brands(self, text: str) -> list[str]:
        matches = self._matches(self.brands, text)
        # Some catalog ``store`` values are generic product words (for
        # example, "Waterproof").  They are not reliable brand evidence.
        matches = [term for term in matches if term not in self.feature_terms and term not in _GENERIC_BRAND_TERMS]
        return [self.brands[term] for term in matches]

    def match_features(self, text: str) -> list[str]:
        ngrams = message_ngrams(text)
        return sorted((term for term in self.feature_terms if term in ngrams), key=lambda item: (-len(item), item))

    @staticmethod
    def _matches(vocabulary: dict[str, str], text: str) -> list[str]:
        ngrams = message_ngrams(text)
        return sorted(
            (term for term in vocabulary if term in ngrams),
            key=lambda item: (-len(item), item),
        )


# These terms are only enabled when they are actually present in the supplied
# catalog.  This prevents a generic language rule from inventing a product
# capability that the catalog cannot express.
FEATURE_CANDIDATES = frozenset(
    {
        "adjustable", "arch support", "breathable", "comfortable", "insulated",
        "lightweight", "machine washable", "non slip", "pockets", "quick dry",
        "reflective", "slip resistant", "stretch", "stretchy", "thermal",
        "water resistant", "waterproof", "wide width", "zipper",
    }
)
_GENERIC_CATEGORIES = {"clothing", "clothing shoes jewelry", "clothing shoes and jewelry", "women", "men", "girls", "boys"}
_GENERIC_BRAND_TERMS = _GENERIC_CATEGORIES | {
    "accessories", "apparel", "bag", "bags", "dress", "dresses", "fashion",
    "jewelry", "shoe", "shoes", "style", "women's", "mens", "men's",
}


@lru_cache(maxsize=4)
def _load_catalog_lexicon(resolved_path: str) -> CatalogLexicon:
    categories: dict[str, str] = {}
    brands: dict[str, str] = {}
    feature_counts: Counter[str] = Counter()
    with Path(resolved_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            for category in product.get("categories") or []:
                term = normalize(category)
                if term and term not in _GENERIC_CATEGORIES and len(term) >= 3:
                    categories.setdefault(term, str(category).strip())
            brand = str(product.get("store") or "").strip()
            normalized_brand = normalize(brand)
            if len(normalized_brand) >= 2:
                brands.setdefault(normalized_brand, brand)
            searchable = " ".join(
                [str(product.get("title") or ""), *map(str, product.get("features") or [])]
            ).lower()
            normalized_searchable = normalize(searchable)
            for term in FEATURE_CANDIDATES:
                if term in normalized_searchable:
                    feature_counts[term] += 1
    return CatalogLexicon(
        categories=categories,
        brands=brands,
        feature_terms=frozenset(feature_counts),
    )
