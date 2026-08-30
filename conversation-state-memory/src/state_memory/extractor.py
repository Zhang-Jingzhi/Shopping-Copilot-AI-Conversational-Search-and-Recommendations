from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import ConstraintType, Intent


@dataclass
class ExtractedSlot:
    name: str
    value: str | float
    constraint_type: ConstraintType
    confidence: float = 0.9


@dataclass
class Extraction:
    intent: Intent
    confidence: float
    slots: list[ExtractedSlot] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    conversion: bool = False
    comparison: bool = False


class RuleBasedExtractor:
    """Small deterministic baseline; replaceable by an LLM/local-model extractor."""

    CATEGORIES = {
        "dress": ["dress", "gown"],
        "shoes": ["shoe", "sneaker", "boot", "heel", "sandal", "loafer"],
        "handbag": ["bag", "handbag", "tote", "purse"],
        "jewelry": ["jewelry", "necklace", "ring", "earring", "bracelet"],
        "top": ["shirt", "top", "blouse", "tee", "t-shirt"],
        "outerwear": ["jacket", "coat", "blazer"],
    }
    COLORS = ("black", "blue", "red", "white", "green", "pink", "purple", "yellow", "brown", "navy", "beige", "grey", "gray")
    OCCASIONS = ("work", "office", "wedding", "party", "date", "casual", "everyday", "running", "gym")
    SOFT_TERMS = ("minimal", "slimming", "comfortable", "trendy", "casual", "formal", "elegant", "vintage")
    MATERIALS = ("cotton", "leather", "denim", "silk", "wool", "linen", "suede")
    GENDERS = {"women": "women", "woman": "women", "men": "men", "man": "men", "girls": "girls", "boys": "boys"}

    def extract(self, utterance: str) -> Extraction:
        text = utterance.lower()
        intent, confidence = self._intent(text)
        slots: list[ExtractedSlot] = []
        rejected: dict[str, list[str]] = {}

        for category, terms in self.CATEGORIES.items():
            if any(re.search(rf"\b{re.escape(term)}s?\b", text) for term in terms):
                slots.append(ExtractedSlot("category", category, ConstraintType.HARD))
                break
        for color in self.COLORS:
            if re.search(rf"\b{color}\b", text):
                target = rejected if re.search(rf"\b(no|not|don't want|do not want)\s+{color}\b", text) else None
                if target is not None:
                    target.setdefault("color", []).append(color)
                else:
                    slots.append(ExtractedSlot("color", "gray" if color == "grey" else color, ConstraintType.HARD))
        for occasion in self.OCCASIONS:
            if re.search(rf"\b{occasion}\b", text):
                slots.append(ExtractedSlot("occasion", "work" if occasion == "office" else occasion, ConstraintType.HARD))
                break
        for term in self.SOFT_TERMS:
            if re.search(rf"\b{term}\b", text):
                slots.append(ExtractedSlot("style", term, ConstraintType.SOFT))
        for material in self.MATERIALS:
            if re.search(rf"\b{material}\b", text):
                slots.append(ExtractedSlot("material", material, ConstraintType.HARD))
                break
        for term, gender in self.GENDERS.items():
            if re.search(rf"\b{term}\b", text):
                slots.append(ExtractedSlot("gender", gender, ConstraintType.HARD))
                break
        size = re.search(r"\bsize\s*(xxs|xs|s|m|l|xl|xxl|\d{1,2})\b", text)
        if size:
            slots.append(ExtractedSlot("size", size.group(1).upper(), ConstraintType.HARD))

        price = re.search(r"(?:under|below|less than|up to|max(?:imum)? of?)\s*\$?\s*(\d+(?:\.\d+)?)", text)
        if price:
            slots.append(ExtractedSlot("price_max", float(price.group(1)), ConstraintType.HARD))
        price = re.search(r"(?:over|above|more than|at least)\s*\$?\s*(\d+(?:\.\d+)?)", text)
        if price:
            slots.append(ExtractedSlot("price_min", float(price.group(1)), ConstraintType.HARD))
        if re.search(r"\b(no|not|without)\s+(high )?heels?\b", text):
            rejected.setdefault("style", []).append("heels")

        return Extraction(
            intent=intent,
            confidence=confidence,
            slots=slots,
            rejected=rejected,
            conversion=bool(re.search(r"\b(i'll take|i will take|buy (it|this|that)|choose (it|this|that)|select (it|this|that))\b", text)),
            comparison=bool(re.search(r"\b(compare|which (one|is)|better between|vs\.?|versus)\b", text)),
        )

    @staticmethod
    def _intent(text: str) -> tuple[Intent, float]:
        if re.search(r"\b(compare|which one|better)\b", text):
            return Intent.COMPARE, 0.9
        if re.search(r"\b(buy|purchase|need|looking for|show me|find me|under|below|budget)\b", text):
            return Intent.BUYING, 0.85
        if re.search(r"\b(ideas|inspiration|trending|popular|browse|explore)\b", text):
            return Intent.BROWSING, 0.8
        return Intent.UNKNOWN, 0.35
