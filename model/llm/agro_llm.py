"""
AgroCulture AI — Core LLM Orchestrator
========================================
The central engine that combines ML models, RAG knowledge base,
rule-based advisors, and a cooperative Traditional LLM backend.

Hybrid architecture:
  1. Intent Classifier → understands WHAT the user wants
  2. Entity Extractor  → pulls out N/P/K, crop, soil, state, etc.
  3. Unprocessable Detector → intercepts out-of-domain / too-vague queries
  4. ML Router → runs sklearn models (crop / fertilizer / yield)
  5. Rule Engine → NPK deficit → fertilizer dosage + cost
  6. RAG KB → FAISS semantic search over 3,870 agri documents
  7. Traditional LLM (Ollama / HuggingFace) → free-form enrichment
     and cooperative fallback for complex agricultural questions
  8. Response Generator → assembles final natural-language answer

Usage:
    from agro_llm import AgroCultureLLM
    llm = AgroCultureLLM()
    response = llm.chat("What crop should I grow with N=80, P=40, K=40?")
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATASET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "dataset"))

CROP_MODEL_PATH = os.path.join(MODEL_BASE, "crop_model", "crop_prediction_model.pkl")
CROP_ENCODER_PATH = os.path.join(MODEL_BASE, "crop_model", "crop_label_encoder.pkl")
FERT_MODEL_PATH = os.path.join(MODEL_BASE, "fartilizer_model", "fertilizer_prediction_model.pkl")
FERT_ENCODER_PATH = os.path.join(MODEL_BASE, "fartilizer_model", "fertilizer_label_encoder.pkl")
YIELD_MODEL_PATH = os.path.join(MODEL_BASE, "state_crop_yeild", "models", "production_model.pkl")
YIELD_META_PATH = os.path.join(MODEL_BASE, "state_crop_yeild", "models", "model_metadata.json")
YIELD_DATASET_PATH = os.path.join(DATASET_DIR, "state_wise_crop_yild.csv")

# Local imports — ensure SCRIPT_DIR is on path so these work from any cwd
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from intent_engine import IntentClassifier, classify_intent, extract_entities, get_missing_fields, format_missing_prompt
from fertilizer_advisor import FertilizerAdvisor, get_recommendation, analyze_soil
from traditional_llm import TraditionalLLMBackend, classify_unprocessable


# ---------------------------------------------------------------------------
# Response Templates
# ---------------------------------------------------------------------------

GREETING_RESPONSES = [
    "🌾 Hello! I'm AgroCulture AI, your intelligent farming assistant.\n"
    "I can help you with:\n"
    "  🌱 Crop prediction — What to grow based on your soil & climate\n"
    "  💊 Fertilizer recommendation — Which fertilizer and how much\n"
    "  📊 Soil analysis — Understand your soil health\n"
    "  🌿 Disease remediation — Identify and treat crop diseases\n"
    "  📈 Yield forecasting — Predict crop production\n\n"
    "Just tell me about your farm conditions and I'll help!\n"
    "Example: 'My soil has N=80, P=40, K=40, pH 6.5, temp 20°C, "
    "humidity 50%, rainfall 75mm in loamy soil. What crop should I grow?'"
]

HELP_TEXT = (
    "📋 How to use AgroCulture AI:\n\n"
    "1️⃣  **Crop Prediction**: Provide soil NPK, temperature, humidity, pH, rainfall, and soil type\n"
    "   Example: 'What crop should I grow with N=80, P=40, K=40, temp 22°C, pH 6.5, "
    "humidity 80%, rainfall 200mm in loamy soil?'\n\n"
    "2️⃣  **Fertilizer Recommendation**: Provide crop name and soil NPK levels\n"
    "   Example: 'What fertilizer for sugarcane? My soil has N=40, P=10, K=80'\n\n"
    "3️⃣  **Soil Analysis**: Provide soil test values\n"
    "   Example: 'Analyze my soil: N=138, P=8.6, K=560, pH=7.46'\n\n"
    "4️⃣  **Disease Query**: Describe symptoms or name a disease\n"
    "   Example: 'How to treat white rot in onion?'\n\n"
    "5️⃣  **Yield Forecast**: Provide state, crop, and year\n"
    "   Example: 'Predict wheat production in Punjab for 2026'\n\n"
    "💡 You can provide values in natural language — I'll extract them automatically!"
)


# ---------------------------------------------------------------------------
# AgroCulture LLM
# ---------------------------------------------------------------------------

class AgroCultureLLM:
    """
    Hybrid AI system for agricultural advisory.

    Combines:
      - 3 scikit-learn ML models (crop prediction, fertilizer, yield forecast)
      - Rule-based NPK deficit → fertilizer dose calculator
      - FAISS RAG knowledge base (3,870 agricultural documents)
      - Traditional LLM backend (Ollama / HuggingFace) for free-form answers
      - Unprocessable question handler with query logging
    """

    def __init__(self, load_kb: bool = True, load_trad_llm: bool = True):
        """
        Initialize the AgroCulture AI system.

        Args:
            load_kb:       Load the FAISS knowledge base (set False for faster startup)
            load_trad_llm: Load the Traditional LLM backend (Ollama / HuggingFace)
        """
        self.intent_classifier = IntentClassifier()
        self.fertilizer_advisor = FertilizerAdvisor()
        self.conversation_history: List[Dict] = []
        self.accumulated_entities: Dict = {}  # Entities accumulated across turns
        self._trad_llm: Optional[TraditionalLLMBackend] = None

        # Model references (lazy-loaded)
        self._crop_model = None
        self._crop_encoder = None
        self._fert_model = None
        self._fert_encoder = None
        self._yield_model = None
        self._yield_meta = None
        self._kb = None

        # Track loaded models
        self.loaded_models = {}

        # Load models
        self._load_models()

        # Load knowledge base
        if load_kb:
            self._load_knowledge_base()

        # Load Traditional LLM backend
        if load_trad_llm:
            self._load_traditional_llm()

    def _load_models(self) -> None:
        """Load all available sklearn models."""
        # Crop prediction model
        if os.path.exists(CROP_MODEL_PATH):
            try:
                self._crop_model = joblib.load(CROP_MODEL_PATH)
                if os.path.exists(CROP_ENCODER_PATH):
                    self._crop_encoder = joblib.load(CROP_ENCODER_PATH)
                self.loaded_models["crop_prediction"] = True
                print("[✓] Crop prediction model loaded")
            except Exception as e:
                print(f"[✗] Crop model failed: {e}")
                self.loaded_models["crop_prediction"] = False
        else:
            print(f"[—] Crop model not found at {CROP_MODEL_PATH}")
            self.loaded_models["crop_prediction"] = False

        # Fertilizer prediction model
        if os.path.exists(FERT_MODEL_PATH):
            try:
                self._fert_model = joblib.load(FERT_MODEL_PATH)
                if os.path.exists(FERT_ENCODER_PATH):
                    self._fert_encoder = joblib.load(FERT_ENCODER_PATH)
                self.loaded_models["fertilizer_prediction"] = True
                print("[✓] Fertilizer prediction model loaded")
            except Exception as e:
                print(f"[✗] Fertilizer model failed: {e}")
                self.loaded_models["fertilizer_prediction"] = False
        else:
            print(f"[—] Fertilizer model not found at {FERT_MODEL_PATH}")
            self.loaded_models["fertilizer_prediction"] = False

        # Yield prediction model
        if os.path.exists(YIELD_MODEL_PATH):
            try:
                self._yield_model = joblib.load(YIELD_MODEL_PATH)
                if os.path.exists(YIELD_META_PATH):
                    with open(YIELD_META_PATH, "r") as f:
                        self._yield_meta = json.load(f)
                self.loaded_models["yield_forecast"] = True
                print("[✓] Yield forecast model loaded")
            except Exception as e:
                print(f"[✗] Yield model failed: {e}")
                self.loaded_models["yield_forecast"] = False
        else:
            print(f"[—] Yield model not found at {YIELD_MODEL_PATH}")
            self.loaded_models["yield_forecast"] = False

    def _load_knowledge_base(self) -> None:
        """Load the RAG knowledge base."""
        try:
            from build_knowledge_base import KnowledgeBase
            self._kb = KnowledgeBase()
            if self._kb.load():
                self.loaded_models["knowledge_base"] = True
            else:
                self.loaded_models["knowledge_base"] = False
        except Exception as e:
            print(f"[—] Knowledge base not available: {e}")
            self.loaded_models["knowledge_base"] = False

    def _load_traditional_llm(self) -> None:
        """Load the Traditional LLM backend (Ollama / HuggingFace)."""
        try:
            self._trad_llm = TraditionalLLMBackend()
            backend = self._trad_llm.backend_name
            self.loaded_models["traditional_llm"] = self._trad_llm.is_llm_available
            if self._trad_llm.is_llm_available:
                print(f"[✓] Traditional LLM loaded ({backend})")
            else:
                print(f"[—] Traditional LLM: template-only mode (Ollama not running, no HF_TOKEN)")
        except Exception as e:
            print(f"[—] Traditional LLM not available: {e}")
            self.loaded_models["traditional_llm"] = False

    # -------------------------------------------------------------------
    # Chat Interface
    # -------------------------------------------------------------------

    def chat(self, message: str) -> str:
        """
        Main chat interface. Process a user message and return a response.

        Args:
            message: User's natural language message

        Returns:
            AI response as a formatted string
        """
        message = message.strip()

        # Store in conversation history
        self.conversation_history.append({"role": "user", "message": message})

        # Check for greetings
        if self._is_greeting(message):
            response = GREETING_RESPONSES[0]
            self.conversation_history.append({"role": "assistant", "message": response})
            return response

        # Check for help
        if self._is_help(message):
            self.conversation_history.append({"role": "assistant", "message": HELP_TEXT})
            return HELP_TEXT

        # ── Step 1: Unprocessable / out-of-domain check ──────────────────
        is_unproc, unproc_reason = classify_unprocessable(message)
        if is_unproc:
            response = self._handle_unprocessable(message, unproc_reason)
            self.conversation_history.append({"role": "assistant", "message": response})
            return response

        # ── Step 2: Intent classification ─────────────────────────────────
        intent, confidence = self.intent_classifier.classify(message)

        # ── Step 3: Entity extraction + accumulation ───────────────────────
        entities = self.intent_classifier.extract_entities(message)
        self.accumulated_entities.update(entities)

        # ── Step 4: Missing field validation ──────────────────────────────
        missing = self.intent_classifier.get_missing_fields(intent, self.accumulated_entities)

        # Too many fields missing AND low confidence → ask for clarification
        if len(missing) > 4 and confidence < 0.5:
            response = format_missing_prompt(missing)
            self.conversation_history.append({"role": "assistant", "message": response})
            return response

        # ── Step 5: Route to ML model / rule engine / RAG ─────────────────
        try:
            if intent == "crop_prediction":
                response = self._handle_crop_prediction(self.accumulated_entities, missing)
            elif intent == "fertilizer_recommendation":
                response = self._handle_fertilizer_recommendation(self.accumulated_entities, missing)
            elif intent == "soil_analysis":
                response = self._handle_soil_analysis(self.accumulated_entities)
            elif intent == "disease_query":
                response = self._handle_disease_query(message)
            elif intent == "yield_forecast":
                response = self._handle_yield_forecast(self.accumulated_entities, missing)
            else:
                # ── Step 6: General QA — try LLM first, fall back to KB ───
                response = self._handle_general_qa(message)
        except Exception as e:
            response = f"⚠️ I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question or check the input values."

        self.conversation_history.append({"role": "assistant", "message": response})
        return response

    def reset(self) -> None:
        """Reset conversation history and accumulated entities."""
        self.conversation_history.clear()
        self.accumulated_entities.clear()

    # -------------------------------------------------------------------
    # Intent Handlers
    # -------------------------------------------------------------------

    def _handle_crop_prediction(self, entities: Dict, missing: List[str]) -> str:
        """Handle crop prediction intent."""
        if not self.loaded_models.get("crop_prediction"):
            return self._fallback_crop_prediction(entities, missing)

        # Check for critical missing fields
        critical_missing = [f for f in missing if f in ["nitrogen", "phosphorus", "potassium"]]
        if critical_missing:
            return format_missing_prompt(missing)

        # Prepare input for the model
        try:
            input_data = pd.DataFrame([{
                'Nitrogen': entities.get('nitrogen', 0),
                'Phosphorus': entities.get('phosphorus', 0),
                'Potassium': entities.get('potassium', 0),
                'Temperature': entities.get('temperature', 25),
                'Humidity': entities.get('humidity', 60),
                'pH_Value': entities.get('ph', 6.5),
                'Rainfall': entities.get('rainfall', 100),
                'Soil_Type': entities.get('soil_type', 'Loamy'),
                'Variety': '',
            }])

            # Predict crop
            prediction = self._crop_model.predict(input_data)

            # Decode if label encoder exists
            if self._crop_encoder is not None:
                try:
                    crop_name = self._crop_encoder.inverse_transform(prediction)[0]
                except Exception:
                    crop_name = str(prediction[0])
            else:
                crop_name = str(prediction[0])

            # Build response
            response = f"🌱 **Crop Prediction Result**\n\n"
            response += f"Based on your soil and climate conditions:\n"
            response += f"  • Nitrogen: {entities.get('nitrogen', 0)}\n"
            response += f"  • Phosphorus: {entities.get('phosphorus', 0)}\n"
            response += f"  • Potassium: {entities.get('potassium', 0)}\n"
            response += f"  • Temperature: {entities.get('temperature', 25)}°C\n"
            response += f"  • Humidity: {entities.get('humidity', 60)}%\n"
            response += f"  • pH: {entities.get('ph', 6.5)}\n"
            response += f"  • Rainfall: {entities.get('rainfall', 100)} mm\n"
            response += f"  • Soil Type: {entities.get('soil_type', 'Loamy')}\n\n"
            response += f"🎯 **Recommended Crop: {crop_name}**\n\n"

            # Add fertilizer recommendation for the predicted crop
            soil_data = {
                "N": entities.get('nitrogen', 0),
                "P": entities.get('phosphorus', 0),
                "K": entities.get('potassium', 0),
                "pH": entities.get('ph', 6.5),
            }
            fert_result = self.fertilizer_advisor.get_full_recommendation(
                soil_data, crop_name
            )
            fert_recs = fert_result.get("fertilizer_recommendations", [])

            if fert_recs and fert_recs[0]["fertilizer"] != "None required":
                response += f"💊 **Fertilizer Suggestion for {crop_name}:**\n"
                for rec in fert_recs:
                    response += f"  • {rec['fertilizer']}: {rec['quantity_kg_per_ha']} kg/ha — {rec['reason']}\n"
                if fert_result.get("total_estimated_cost_per_ha", 0) > 0:
                    response += f"  💰 Est. Cost: ₹{fert_result['total_estimated_cost_per_ha']}/hectare\n"
            else:
                response += f"✅ Your soil is already well-balanced for {crop_name}!\n"

            # Enrich with knowledge base
            kb_context = self._search_kb(f"{crop_name} cultivation requirements", top_k=2)
            if kb_context:
                response += f"\n📚 **Additional Information:**\n"
                for ctx in kb_context:
                    response += f"  • {ctx['text'][:200]}\n"

            # Clear accumulated entities for fresh start
            self.accumulated_entities.clear()

            return response

        except Exception as e:
            return f"⚠️ Error in crop prediction: {str(e)}\n\nPlease verify your input values and try again."

    def _fallback_crop_prediction(self, entities: Dict, missing: List[str]) -> str:
        """Fallback crop prediction using knowledge base when model isn't available."""
        soil_type = entities.get("soil_type", "")
        n = entities.get("nitrogen", 0)
        p = entities.get("phosphorus", 0)
        k = entities.get("potassium", 0)

        query = f"crop suitable for {soil_type} soil with nitrogen {n} phosphorus {p} potassium {k}"
        kb_results = self._search_kb(query, top_k=5, category="crop_requirements")

        if kb_results:
            response = "🌱 **Crop Suggestions** (from knowledge base):\n\n"
            for i, r in enumerate(kb_results, 1):
                response += f"  {i}. {r['text'][:200]}\n\n"
            response += "\n⚠️ Note: ML model not available. Using knowledge base for suggestions."
            return response

        if missing:
            return format_missing_prompt(missing)

        return "I couldn't find suitable crop recommendations. Please provide more details about your soil conditions."

    def _handle_fertilizer_recommendation(self, entities: Dict, missing: List[str]) -> str:
        """Handle fertilizer recommendation intent."""
        crop = entities.get("crop", "")

        # If we have crop and soil NPK, use the rule-based advisor
        has_npk = all(k in entities for k in ["nitrogen", "phosphorus", "potassium"])

        if crop and has_npk:
            soil_data = {
                "N": entities.get("nitrogen", 0),
                "P": entities.get("phosphorus", 0),
                "K": entities.get("potassium", 0),
                "pH": entities.get("ph"),
            }
            area = entities.get("area", 1.0)

            response = get_recommendation(soil_data, crop, area)

            # Also try ML model prediction
            if self.loaded_models.get("fertilizer_prediction"):
                try:
                    ml_pred = self._predict_fertilizer_ml(entities)
                    if ml_pred:
                        response += f"\n\n🤖 **ML Model Prediction:** {ml_pred}"
                except Exception:
                    pass

            # Enrich with KB
            kb_context = self._search_kb(f"fertilizer for {crop}", top_k=2)
            if kb_context:
                response += f"\n\n📚 **Related Knowledge:**\n"
                for ctx in kb_context:
                    response += f"  • {ctx['text'][:200]}\n"

            self.accumulated_entities.clear()
            return response

        # If we have enough for ML model
        if self.loaded_models.get("fertilizer_prediction") and crop:
            ml_pred = self._predict_fertilizer_ml(entities)
            if ml_pred:
                response = f"💊 **Fertilizer Recommendation**\n\n"
                response += f"For {crop}: **{ml_pred}**\n\n"

                if has_npk:
                    soil_data = {
                        "N": entities.get("nitrogen", 0),
                        "P": entities.get("phosphorus", 0),
                        "K": entities.get("potassium", 0),
                    }
                    response += get_recommendation(soil_data, crop)

                self.accumulated_entities.clear()
                return response

        # Ask for missing fields
        essential_missing = []
        if not crop:
            essential_missing.append("crop")
        if not has_npk:
            for n in ["nitrogen", "phosphorus", "potassium"]:
                if n not in entities:
                    essential_missing.append(n)

        if essential_missing:
            return format_missing_prompt(essential_missing)

        return "Please provide your crop name and soil nutrient levels (N, P, K) for a fertilizer recommendation."

    def _predict_fertilizer_ml(self, entities: Dict) -> Optional[str]:
        """Use the ML model to predict fertilizer."""
        if not self._fert_model:
            return None

        try:
            input_data = pd.DataFrame([{
                'Temparature': entities.get('temperature', 25),
                'Humidity': entities.get('humidity', 60),
                'Moisture': entities.get('moisture', 40),
                'Soil_Type': entities.get('soil_type', 'Loamy'),
                'Crop_Type': entities.get('crop', 'Rice'),
                'Nitrogen': entities.get('nitrogen', 50),
                'Potassium': entities.get('potassium', 40),
                'Phosphorous': entities.get('phosphorus', 40),
            }])

            prediction = self._fert_model.predict(input_data)

            if self._fert_encoder is not None:
                try:
                    return self._fert_encoder.inverse_transform(prediction)[0]
                except Exception:
                    return str(prediction[0])
            return str(prediction[0])

        except Exception:
            return None

    def _handle_soil_analysis(self, entities: Dict) -> str:
        """Handle soil analysis intent."""
        soil_data = {}
        for key_map in [("nitrogen", "N"), ("phosphorus", "P"), ("potassium", "K"),
                        ("ph", "pH")]:
            entity_key, soil_key = key_map
            if entity_key in entities:
                soil_data[soil_key] = entities[entity_key]

        if not soil_data:
            return format_missing_prompt(["nitrogen", "phosphorus", "potassium", "ph"])

        analysis = self.fertilizer_advisor.analyze_soil(soil_data)

        response = "📊 **Soil Health Analysis**\n\n"

        for nutrient, info in analysis.items():
            status_emoji = {
                "low": "🔴", "medium": "🟡", "high": "🟢",
                "acidic": "🔴", "slightly_acidic": "🟡", "neutral": "🟢",
                "slightly_alkaline": "🟡", "alkaline": "🔴",
            }.get(info["status"], "⚪")
            response += f"  {status_emoji} **{nutrient}**: {info['value']} — {info['status'].replace('_', ' ').title()}\n"

        # Overall assessment
        nutrient_statuses = [v["status"] for k, v in analysis.items() if k in ["N", "P", "K"]]
        if all(s in ["high", "medium"] for s in nutrient_statuses):
            response += "\n✅ **Overall: Your soil has good nutrient levels!**\n"
        elif any(s == "low" for s in nutrient_statuses):
            low_nutrients = [k for k, v in analysis.items() if k in ["N", "P", "K"] and v["status"] == "low"]
            response += f"\n⚠️ **Attention: Low levels detected for {', '.join(low_nutrients)}.**\n"
            response += "Consider applying appropriate fertilizers to improve soil fertility.\n"

        # pH recommendations
        if "pH" in analysis:
            ph_status = analysis["pH"]["status"]
            if ph_status == "acidic":
                response += "\n🔧 **pH Action:** Soil is too acidic. Apply lime (calcium carbonate) to raise pH.\n"
            elif ph_status == "alkaline":
                response += "\n🔧 **pH Action:** Soil is too alkaline. Apply gypsum or sulfur to lower pH.\n"

        # KB enrichment
        query = f"soil fertility analysis nitrogen {soil_data.get('N', '')} phosphorus {soil_data.get('P', '')}"
        kb_results = self._search_kb(query, top_k=2, category="soil_analysis")
        if kb_results:
            response += "\n📚 **Similar Soil Profiles from Database:**\n"
            for r in kb_results:
                response += f"  • {r['text'][:200]}\n"

        return response

    def _handle_disease_query(self, message: str) -> str:
        """Handle disease-related queries using the knowledge base."""
        response = "🌿 **Crop Disease Information**\n\n"

        # Search knowledge base
        kb_results = self._search_kb(message, top_k=5, category="disease_info")

        if kb_results:
            response += "Here's what I found:\n\n"
            for i, r in enumerate(kb_results, 1):
                text = r['text'][:300]
                score = r['score']
                crop = r.get('crop', '')
                disease = r.get('disease', '')

                header_parts = []
                if crop:
                    header_parts.append(crop)
                if disease:
                    header_parts.append(disease)
                header = " — ".join(header_parts) if header_parts else f"Result {i}"

                response += f"**{i}. {header}** (relevance: {score:.0%})\n"
                response += f"   {text}\n\n"
        else:
            # Fallback: try general search
            kb_results = self._search_kb(message, top_k=3)
            if kb_results:
                response += "I found some related information:\n\n"
                for i, r in enumerate(kb_results, 1):
                    response += f"  {i}. {r['text'][:200]}\n\n"
            else:
                response += ("I couldn't find specific information about this disease in my database.\n\n"
                           "💡 **General Tips:**\n"
                           "  • Maintain proper crop spacing for air circulation\n"
                           "  • Use disease-resistant varieties when available\n"
                           "  • Practice crop rotation to break disease cycles\n"
                           "  • Apply appropriate fungicides/pesticides as a last resort\n"
                           "  • Consult your local agriculture extension officer for specific treatments")

        return response

    def _handle_yield_forecast(self, entities: Dict, missing: List[str]) -> str:
        """Handle yield forecast intent."""
        state = entities.get("state", "")
        crop = entities.get("crop", "")
        year = entities.get("year", 2026)

        if not state or not crop:
            need = []
            if not state:
                need.append("state")
            if not crop:
                need.append("crop")
            return format_missing_prompt(need)

        if not self.loaded_models.get("yield_forecast"):
            return self._fallback_yield_forecast(entities)

        try:
            # Get historical defaults
            from model.state_crop_yeild.test_predict import get_historical_defaults
            defaults = get_historical_defaults(YIELD_DATASET_PATH, state, crop)

            if defaults is None:
                # Try without exact match — search KB instead
                return self._fallback_yield_forecast(entities)

            season = entities.get("season", defaults.get("Season", "Whole Year"))
            area = entities.get("area", defaults.get("Area", 1000))
            rainfall = entities.get("rainfall", defaults.get("Annual_Rainfall", 1000))
            fertilizer = defaults.get("Fertilizer", 50000)
            pesticide = defaults.get("Pesticide", 1000)

            input_df = pd.DataFrame([{
                "State": state,
                "Crop": crop,
                "Season": season,
                "Crop_Year": year,
                "Area": area,
                "Annual_Rainfall": rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide,
            }])

            prediction = max(0, self._yield_model.predict(input_df)[0])
            estimated_yield = prediction / area if area > 0 else 0

            response = f"📈 **Crop Production Forecast**\n\n"
            response += f"  State: {state}\n"
            response += f"  Crop: {crop}\n"
            response += f"  Season: {season}\n"
            response += f"  Year: {year}\n"
            response += f"  Area: {area:.0f} hectares\n\n"
            response += f"🎯 **Predicted Production: {prediction:,.2f} tonnes**\n"
            response += f"   Estimated Yield: {estimated_yield:.4f} tonnes/hectare\n"

            if defaults:
                response += f"\n📊 Historical defaults used (based on years {defaults.get('based_on_years', [])}):\n"
                response += f"   Rainfall: {rainfall:.0f} mm, Fertilizer: {fertilizer:.0f}, Pesticide: {pesticide:.0f}\n"

            self.accumulated_entities.clear()
            return response

        except Exception as e:
            return self._fallback_yield_forecast(entities)

    def _fallback_yield_forecast(self, entities: Dict) -> str:
        """Fallback yield forecast using knowledge base."""
        state = entities.get("state", "")
        crop = entities.get("crop", "")

        query = f"{crop} production yield in {state}"
        kb_results = self._search_kb(query, top_k=3)

        response = f"📈 **Yield Information for {crop} in {state}**\n\n"

        if kb_results:
            for i, r in enumerate(kb_results, 1):
                response += f"  {i}. {r['text'][:250]}\n\n"
        else:
            response += "I don't have enough data for a precise forecast.\n"
            response += "Please ensure the yield forecast model has been trained.\n"

        return response

    def _handle_general_qa(self, message: str) -> str:
        """
        Handle general agricultural queries.

        Strategy:
          1. Search RAG knowledge base — return if high-confidence hit found
          2. If KB score is low (< 0.5), escalate to Traditional LLM backend
          3. If LLM unavailable, return KB results even at lower confidence
          4. Final fallback: suggest specific question formats
        """
        kb_results = self._search_kb(message, top_k=5)
        top_score = kb_results[0]["score"] if kb_results else 0.0

        # High-confidence KB hit — return KB results
        if kb_results and top_score > 0.50:
            response = "📚 **Here's what I found:**\n\n"
            for i, r in enumerate(kb_results, 1):
                text = r['text'][:250]
                score = r['score']
                source = r.get('source', 'knowledge base')
                response += f"  **{i}.** {text}\n"
                response += f"     _(Source: {source}, Relevance: {score:.0%})_\n\n"
            return response

        # Low-confidence KB — try Traditional LLM
        if self._trad_llm:
            llm_response = self._trad_llm.generate(
                user_message=message,
                context="",
                conversation_history=self.conversation_history,
            )
            if llm_response and len(llm_response) > 40:
                # If KB had at least something, append it as context
                if kb_results and top_score > 0.25:
                    kb_snippet = kb_results[0]["text"][:200]
                    return (
                        f"🤖 **AgroCulture AI** (LLM):\n\n{llm_response}\n\n"
                        f"📚 **Related KB Context:** {kb_snippet}..."
                    )
                return f"🤖 **AgroCulture AI** (LLM):\n\n{llm_response}"

        # Medium KB results exist even if below threshold — show them
        if kb_results and top_score > 0.25:
            response = "📚 **Here's what I found** (partial match):\n\n"
            for i, r in enumerate(kb_results[:3], 1):
                text = r['text'][:200]
                score = r['score']
                response += f"  **{i}.** {text}\n"
                response += f"     _(Relevance: {score:.0%})_\n\n"
            return response

        # Nothing useful — guide the user
        return (
            "🤔 I'm not sure how to answer that specific question. "
            "I specialise in:\n\n"
            "  🌱 **Crop prediction** — 'What crop for my soil conditions?'\n"
            "  💊 **Fertilizer advice** — 'What fertilizer for rice?'\n"
            "  📊 **Soil analysis** — 'Analyze my soil: N=80, P=40, K=40'\n"
            "  🌿 **Disease info** — 'How to treat blight in potato?'\n"
            "  📈 **Yield forecast** — 'Predict rice production in Bihar'\n\n"
            "Try rephrasing your question or providing specific values!"
        )

    # -------------------------------------------------------------------
    # Unprocessable Handler
    # -------------------------------------------------------------------

    def _handle_unprocessable(self, message: str, reason: str) -> str:
        """
        Dedicated handler for questions the main orchestrator cannot process.

        Routes to TraditionalLLMBackend.handle_unprocessable() which:
          - Logs the query to data/unprocessable_log.jsonl
          - Returns a polite out-of-domain decline, clarification request,
            or LLM-generated agricultural answer (for border cases)
        """
        if self._trad_llm:
            return self._trad_llm.handle_unprocessable(
                message, reason, self.conversation_history
            )

        # trad_llm not loaded — minimal response
        reason_map = {
            "out_of_domain": (
                "🌾 I'm specialised in agriculture — that topic is outside my expertise.\n"
                "Try asking about crops, fertilizers, soil, diseases, or yield forecasting!"
            ),
            "too_vague": (
                "🤔 Could you be more specific? Tell me about your crop, soil, or farming problem."
            ),
            "border_agri": (
                "📞 That's an interesting agricultural question! For specialised advice, "
                "contact **Kisan Call Centre: 1800-180-1551** (free, 24×7)."
            ),
        }
        return reason_map.get(reason, "🌾 I couldn't process that question. Please try rephrasing.")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _search_kb(self, query: str, top_k: int = 3,
                   category: Optional[str] = None) -> List[Dict]:
        """Search knowledge base safely."""
        if self._kb and self._kb.is_loaded:
            try:
                return self._kb.search(query, top_k=top_k, category=category)
            except Exception:
                return []
        return []

    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting."""
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon",
                     "good evening", "namaste", "namaskar", "start",
                     "help me", "what can you do"]
        msg_lower = message.lower().strip()
        return any(msg_lower.startswith(g) or msg_lower == g for g in greetings)

    def _is_help(self, message: str) -> bool:
        """Check if message is asking for help."""
        help_words = ["/help", "how to use", "how do i use", "instructions",
                      "guide", "tutorial", "usage"]
        msg_lower = message.lower().strip()
        return any(h in msg_lower for h in help_words)

    @property
    def model_status(self) -> str:
        """Get a formatted string of model loading status."""
        lines = ["📦 Model Status:"]
        status_map = {
            "crop_prediction":      "Crop Prediction (sklearn)",
            "fertilizer_prediction": "Fertilizer Prediction (sklearn)",
            "yield_forecast":        "Yield Forecast (sklearn)",
            "knowledge_base":        "Knowledge Base (FAISS RAG)",
            "traditional_llm":       "Traditional LLM (Ollama/HuggingFace)",
        }
        for key, name in status_map.items():
            loaded = self.loaded_models.get(key, False)
            icon = "✅" if loaded else "—"
            lines.append(f"  {icon} {name}")

        # Show which LLM backend is active
        if self._trad_llm:
            lines.append(f"     └─ LLM backend: {self._trad_llm.backend_name}")
            if self._trad_llm.is_llm_available:
                lines.append(f"        model: {self._trad_llm.ollama.model_name if self._trad_llm.ollama.is_available else self._trad_llm.hf.model_name}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  AgroCulture AI — Quick Test")
    print("=" * 70)

    llm = AgroCultureLLM(load_kb=True)
    print("\n" + llm.model_status)

    test_messages = [
        "Hello",
        "What crop should I grow with N=80, P=40, K=40, temperature 20°C, humidity 50%, pH 6.5, rainfall 75mm in loamy soil?",
        "What fertilizer should I use for sugarcane? My soil has N=40, P=10, K=80",
        "Analyze my soil: N=138, P=8.6, K=560, pH=7.46",
    ]

    for msg in test_messages:
        print(f"\n{'=' * 70}")
        print(f"USER: {msg}")
        print(f"{'=' * 70}")
        response = llm.chat(msg)
        print(response)
        llm.reset()
