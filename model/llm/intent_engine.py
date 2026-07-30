"""
Intent Classifier & Entity Extractor for AgroCulture AI
========================================================
Classifies user messages into agricultural intents and extracts
relevant entities (NPK values, crop names, soil types, etc.)
using regex + keyword matching.

Supported Intents:
    - crop_prediction
    - fertilizer_recommendation
    - soil_analysis
    - disease_query
    - yield_forecast
    - general_qa (fallback)
"""

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Intent Definitions
# ---------------------------------------------------------------------------

INTENT_PATTERNS = {
    "crop_prediction": {
        "keywords": [
            "what crop", "which crop", "suggest crop", "suggest a crop",
            "best crop", "recommend crop", "recommend a crop", "crop suggestion",
            "grow what", "what should i grow", "what to grow", "what can i grow",
            "suitable crop", "ideal crop", "crop for my", "predict crop",
            "crop prediction", "crop recommend",
        ],
        "patterns": [
            r"(?:what|which|best|suitable|ideal)\s+(?:crop|plant|vegetable|grain)",
            r"(?:suggest|recommend)\s+(?:a\s+)?(?:crop|plant)",
            r"what\s+(?:should|can|could)\s+i\s+(?:grow|plant|cultivate)",
            r"crop\s+(?:for|suitable|prediction|suggest|recommend)",
        ],
    },
    "fertilizer_recommendation": {
        "keywords": [
            "fertilizer", "fertiliser", "which fertilizer", "what fertilizer",
            "suggest fertilizer", "recommend fertilizer", "fertilizer for",
            "npk", "nutrient", "nutrient deficiency", "urea", "dap",
            "how much fertilizer", "fertilizer dose", "fertilizer dosage",
            "fertilizer quantity", "nutrient recommendation",
            "what should i apply", "soil nutrient",
        ],
        "patterns": [
            r"(?:what|which|best|recommend|suggest)\s+(?:a\s+)?fertili[sz]er",
            r"fertili[sz]er\s+(?:for|recommendation|suggest|dose|dosage|quantity)",
            r"(?:npk|nutrient)\s+(?:deficiency|deficit|requirement|recommendation)",
            r"how\s+much\s+(?:fertili[sz]er|urea|dap|mop|potash)",
            r"(?:need|require|apply)\s+(?:.*\s+)?fertili[sz]er",
        ],
    },
    "soil_analysis": {
        "keywords": [
            "soil health", "soil fertility", "soil test", "soil quality",
            "soil analysis", "soil component", "soil condition", "soil report",
            "soil status", "soil check", "my soil", "analyze soil",
            "soil ph", "organic carbon", "organic matter",
            "is my soil", "soil good", "soil bad",
        ],
        "patterns": [
            r"(?:soil|land)\s+(?:health|fertility|test|quality|analysis|condition|report|status|component)",
            r"(?:analyze|analyse|check|assess)\s+(?:my\s+)?soil",
            r"(?:is|how)\s+(?:my\s+)?soil",
            r"soil\s+(?:ph|nitrogen|phosphorus|potassium|organic)",
        ],
    },
    "disease_query": {
        "keywords": [
            "disease", "pest", "infection", "blight", "rot", "wilt",
            "rust", "mildew", "fungus", "fungal", "bacterial", "virus",
            "viral", "insect", "remedy", "cure", "treatment", "pesticide",
            "how to treat", "affected", "damaged", "spots on",
            "yellow leaves", "wilting", "dying", "brown spots",
        ],
        "patterns": [
            r"(?:disease|pest|infection|blight|rot|wilt|rust|mildew|fungus)",
            r"(?:remedy|cure|treatment|treat|manage)\s+(?:for\s+)?(?:.*\s+)?(?:disease|pest|infection)",
            r"(?:yellow|brown|black)\s+(?:leaf|leaves|spot|spots)",
            r"(?:plant|crop|leaf|leaves)\s+(?:is|are)\s+(?:dying|wilting|affected|damaged|infected)",
            r"how\s+to\s+(?:treat|cure|manage|control)\s+",
        ],
    },
    "yield_forecast": {
        "keywords": [
            "yield", "production", "forecast", "how much will",
            "predict yield", "predict production", "expected yield",
            "expected production", "crop yield", "crop production",
            "harvest", "output", "tonnes", "quintal",
            "state wise", "state production",
        ],
        "patterns": [
            r"(?:predict|forecast|estimate|expected)\s+(?:.*\s+)?(?:yield|production|harvest|output)",
            r"(?:yield|production)\s+(?:for|in|of|prediction|forecast)",
            r"how\s+much\s+(?:will|can|could)\s+(?:i\s+)?(?:get|produce|harvest)",
            r"(?:crop|state)\s+(?:wise\s+)?(?:yield|production)",
        ],
    },
}

# ---------------------------------------------------------------------------
# Entity Extraction Patterns
# ---------------------------------------------------------------------------

NUMERIC_ENTITIES = {
    "nitrogen": [
        r"(?:nitrogen|N)\s*[=:]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*(?:nitrogen)",
        r"nitrogen\s+(?:is\s+)?(\d+\.?\d*)",
    ],
    "phosphorus": [
        r"(?:phosphorus|phosphorous|P)\s*[=:]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*(?:phosphorus|phosphorous)",
        r"phosph(?:orus|orous)\s+(?:is\s+)?(\d+\.?\d*)",
    ],
    "potassium": [
        r"(?:potassium|K)\s*[=:]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*(?:potassium)",
        r"potassium\s+(?:is\s+)?(\d+\.?\d*)",
    ],
    "temperature": [
        r"(?:temperature|temp)\s*[=:]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*°?\s*[cC]",
        r"temperature\s+(?:is\s+)?(\d+\.?\d*)",
        r"temp\s+(\d+\.?\d*)",
    ],
    "humidity": [
        r"(?:humidity|hum)\s*[=:]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*%\s*humidity",
        r"humidity\s+(?:is\s+)?(\d+\.?\d*)",
    ],
    "ph": [
        r"(?:ph|pH)\s*[=:]\s*(\d+\.?\d*)",
        r"ph\s+(?:is\s+)?(\d+\.?\d*)",
        r"(?:ph|pH)\s+(?:value\s+)?(?:of\s+)?(\d+\.?\d*)",
    ],
    "rainfall": [
        r"(?:rainfall|rain)\s*[=:]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*mm\s*(?:rain|rainfall)?",
        r"rainfall\s+(?:is\s+)?(\d+\.?\d*)",
    ],
    "moisture": [
        r"(?:moisture)\s*[=:]\s*(\d+\.?\d*)",
        r"moisture\s+(?:is\s+)?(\d+\.?\d*)",
    ],
    "area": [
        r"(?:area)\s*[=:]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*(?:hectares?|acres?|ha\b)",
        r"area\s+(?:is\s+)?(\d+\.?\d*)",
    ],
}

# Entities that need CASE-SENSITIVE matching (N, P, K use uppercase single letters)
CASE_SENSITIVE_ENTITIES = {"nitrogen", "phosphorus", "potassium"}

SOIL_TYPES = [
    "sandy", "loamy", "clay", "clayey", "red", "black",
    "alluvial", "laterite", "silt", "silty", "peaty", "saline",
    "red and yellow", "alluvial soils", "laterite soils", "black soils",
]

CROP_NAMES = [
    "rice", "wheat", "maize", "corn", "sugarcane", "cotton", "jute",
    "tea", "coffee", "rubber", "coconut", "groundnut", "soybean",
    "sunflower", "mustard", "sesame", "linseed", "castor", "tobacco",
    "potato", "onion", "tomato", "brinjal", "cabbage", "cauliflower",
    "pea", "peas", "beans", "lentil", "chickpea", "gram", "arhar",
    "moong", "urad", "bajra", "jowar", "ragi", "barley", "oat",
    "millet", "paddy", "arecanut", "banana", "mango", "apple",
    "orange", "grapes", "watermelon", "papaya", "pepper", "chilli",
    "turmeric", "ginger", "garlic", "coriander", "cumin",
    "cardamom", "clove", "nutmeg", "tapioca", "kulthi",
]

INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar",
    "chhattisgarh", "goa", "gujarat", "haryana", "himachal pradesh",
    "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra",
    "manipur", "meghalaya", "mizoram", "nagaland", "odisha",
    "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana",
    "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "andaman and nicobar", "chandigarh", "delhi",
    "jammu and kashmir", "ladakh", "puducherry",
]

SEASONS = ["kharif", "rabi", "whole year", "summer", "winter", "zaid", "autumn"]


# ---------------------------------------------------------------------------
# Intent Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Classifies user messages into agricultural intents and extracts entities.
    Uses a scoring system: keyword matches + regex pattern matches.
    """

    def classify(self, message: str) -> Tuple[str, float]:
        """
        Classify the intent of a user message.

        Returns:
            (intent_name, confidence_score)
        """
        message_lower = message.lower().strip()
        scores: Dict[str, float] = {}

        for intent, config in INTENT_PATTERNS.items():
            score = 0.0

            # Keyword matching
            for keyword in config["keywords"]:
                if keyword in message_lower:
                    score += 1.0

            # Regex pattern matching (weighted higher)
            for pattern in config["patterns"]:
                if re.search(pattern, message_lower):
                    score += 1.5

            scores[intent] = score

        if not scores or max(scores.values()) == 0:
            return "general_qa", 0.0

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # Normalize confidence (cap at 1.0)
        confidence = min(best_score / 5.0, 1.0)

        return best_intent, confidence

    def extract_entities(self, message: str) -> Dict:
        """
        Extract all recognizable entities from the message.

        Returns dict with keys like:
            nitrogen, phosphorus, potassium, temperature, humidity,
            ph, rainfall, moisture, area, soil_type, crop, state, season
        """
        message_lower = message.lower().strip()
        entities: Dict = {}

        # Extract numeric entities
        for entity_name, patterns in NUMERIC_ENTITIES.items():
            # For N, P, K — match on ORIGINAL message (case-sensitive for uppercase letters)
            # For others — match on lowercase message
            if entity_name in CASE_SENSITIVE_ENTITIES:
                search_text = message.strip()
            else:
                search_text = message_lower

            for pattern in patterns:
                if entity_name in CASE_SENSITIVE_ENTITIES:
                    match = re.search(pattern, search_text)
                else:
                    match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    try:
                        entities[entity_name] = float(match.group(1))
                    except (ValueError, IndexError):
                        continue
                    break

        # Extract soil type (use word boundaries to avoid false matches like 'predict' -> 'red')
        for soil in sorted(SOIL_TYPES, key=len, reverse=True):
            pattern = r'\b' + re.escape(soil) + r'\b'
            if re.search(pattern, message_lower):
                entities["soil_type"] = soil.title()
                break

        # Extract crop name
        for crop in sorted(CROP_NAMES, key=len, reverse=True):
            pattern = r'\b' + re.escape(crop) + r'\b'
            if re.search(pattern, message_lower):
                entities["crop"] = crop.title()
                break

        # Extract state
        for state in sorted(INDIAN_STATES, key=len, reverse=True):
            if state in message_lower:
                entities["state"] = state.title()
                break

        # Extract season
        for season in SEASONS:
            if season in message_lower:
                entities["season"] = season.title()
                break

        # Extract year
        year_match = re.search(r'\b(20[0-9]{2})\b', message)
        if year_match:
            entities["year"] = int(year_match.group(1))

        return entities

    def get_missing_fields(self, intent: str, entities: Dict) -> List[str]:
        """
        Determine which required fields are missing for a given intent.
        """
        required_fields = {
            "crop_prediction": [
                "nitrogen", "phosphorus", "potassium", "temperature",
                "humidity", "ph", "rainfall", "soil_type",
            ],
            "fertilizer_recommendation": [
                "temperature", "humidity", "moisture", "soil_type",
                "crop", "nitrogen", "potassium", "phosphorus",
            ],
            "soil_analysis": [
                "nitrogen", "phosphorus", "potassium", "ph",
            ],
            "yield_forecast": [
                "state", "crop",
            ],
            "disease_query": [],
            "general_qa": [],
        }

        needed = required_fields.get(intent, [])
        missing = [f for f in needed if f not in entities]
        return missing

    def format_missing_prompt(self, missing: List[str]) -> str:
        """
        Generate a friendly prompt asking for missing information.
        """
        if not missing:
            return ""

        field_names = {
            "nitrogen": "Nitrogen (N) level",
            "phosphorus": "Phosphorus (P) level",
            "potassium": "Potassium (K) level",
            "temperature": "Temperature (°C)",
            "humidity": "Humidity (%)",
            "ph": "Soil pH value",
            "rainfall": "Rainfall (mm)",
            "moisture": "Moisture level",
            "soil_type": f"Soil type ({', '.join(SOIL_TYPES[:6])})",
            "crop": "Crop name",
            "state": "State name",
            "season": "Season (Kharif/Rabi/Whole Year)",
            "area": "Area (hectares)",
            "year": "Year",
        }

        lines = ["I need a few more details to give you an accurate answer:\n"]
        for field in missing:
            name = field_names.get(field, field.replace("_", " ").title())
            lines.append(f"  • {name}")

        lines.append("\nPlease provide these values (e.g., 'N=80, P=40, K=40, pH=6.5').")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_classifier = IntentClassifier()

def classify_intent(message: str) -> Tuple[str, float]:
    """Classify intent of a message. Returns (intent, confidence)."""
    return _classifier.classify(message)

def extract_entities(message: str) -> Dict:
    """Extract entities from a message."""
    return _classifier.extract_entities(message)

def get_missing_fields(intent: str, entities: Dict) -> List[str]:
    """Get missing required fields for an intent."""
    return _classifier.get_missing_fields(intent, entities)

def format_missing_prompt(missing: List[str]) -> str:
    """Format a prompt asking for missing fields."""
    return _classifier.format_missing_prompt(missing)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_messages = [
        "What crop should I grow if my soil has N=80, P=40, K=40, temperature 20°C, humidity 50%, pH 6.5, rainfall 75mm in loamy soil?",
        "Suggest a fertilizer for rice in clayey soil with temperature 25, humidity 84, moisture 32",
        "My soil test shows N=138, P=8.6, K=560, pH=7.46. Is my soil fertile?",
        "How to treat white rot disease in onion?",
        "Predict wheat production in Punjab for 2026",
        "Tell me about organic farming",
    ]

    print("=" * 70)
    print("  Intent Engine Test")
    print("=" * 70)

    for msg in test_messages:
        intent, conf = classify_intent(msg)
        entities = extract_entities(msg)
        missing = get_missing_fields(intent, entities)

        print(f"\nQ: {msg}")
        print(f"   Intent: {intent} (confidence: {conf:.2f})")
        print(f"   Entities: {entities}")
        if missing:
            print(f"   Missing: {missing}")
        print("-" * 70)
