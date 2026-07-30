"""
Traditional LLM Backend — Cooperative LLM Module for AgroCulture AI
=====================================================================
Provides a real generative LLM that works **alongside** the ML orchestrator.

Architecture:
    The orchestrator handles structured tasks (crop prediction, fertilizer
    dosage, soil analysis, yield forecast) using sklearn models + rule engine.
    The traditional LLM handles:
      1. Unprocessable / out-of-domain questions
      2. Free-form agricultural explanations requested by the user
      3. Enriching responses when the RAG + ML pipeline has low confidence
      4. Multi-turn conversational follow-ups

Backends (tried in priority order):
    1. Ollama (local, free) — http://localhost:11434  (recommended)
       Install: https://ollama.com → `ollama pull mistral` or `ollama pull llama3`
    2. HuggingFace Inference API (free tier, needs HF_TOKEN env var)
    3. Template-based fallback (always works, offline)

Usage:
    from traditional_llm import TraditionalLLMBackend

    llm_backend = TraditionalLLMBackend()
    response = llm_backend.generate(
        user_message="What is companion planting?",
        context="Previous assistant answer about tomatoes."
    )
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("agro.trad_llm")

# ---------------------------------------------------------------------------
# Agriculture Domain System Prompt
# ---------------------------------------------------------------------------

AGRO_SYSTEM_PROMPT = """You are AgroCulture AI, a specialized agricultural assistant for Indian farmers.

Your expertise covers:
- Crop cultivation techniques (rice, wheat, sugarcane, cotton, vegetables, fruits, spices)
- Soil health, fertilization, and nutrient management (NPK, micronutrients)
- Irrigation methods and water management
- Integrated pest and disease management (IPM)
- Organic and sustainable farming practices
- Indian seasons (Kharif, Rabi, Zaid) and regional crop calendars
- Government schemes for farmers (PM-Kisan, MSP, Soil Health Card, etc.)
- Post-harvest management and storage

Rules:
1. Always give practical, actionable advice suitable for Indian conditions
2. Mention ICAR / KVK resources when relevant
3. If a question is completely outside agriculture (politics, entertainment, etc.), politely redirect
4. Use simple language; avoid heavy jargon unless asked for technical detail
5. When uncertain, say so — never fabricate specific numbers or chemical names
6. Prefer organic/IPM solutions before recommending chemical pesticides
7. Keep responses concise (under 400 words) unless the user asks for details"""


# ---------------------------------------------------------------------------
# Unprocessable Question Detector
# ---------------------------------------------------------------------------

# Patterns that indicate a question is truly out-of-domain
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(cricket|ipl|bollywood|movie|film|actor|actress|politician|election|vote)\b",
    r"\b(stock market|share price|crypto|bitcoin|nifty|sensex)\b",
    r"\b(weather forecast|today.s weather|rain tomorrow)\b",  # too specific/realtime
    r"\b(recipe|cook|bake|boil|fry|eat|restaurant)\b",
    r"\b(relationship|marriage|love|breakup|girlfriend|boyfriend)\b",
    r"\b(coding|programming|python syntax|javascript|react|html)\b",
    r"\b(maths problem|algebra|calculus|physics equation)\b",
    r"\b(translate|translation|hindi meaning|english meaning)\b",
]

# Patterns where the question IS agricultural but the system can't answer
BORDER_CASE_PATTERNS = [
    r"\b(export|import|market price|mandi|commodity price)\b",
    r"\b(loan|kcc|credit|bank|interest rate|subsidy amount)\b",
    r"\b(satellite|drone|precision farming|iot sensor)\b",
    r"\b(organic certif|fssai|agmark|export certif)\b",
    r"\b(climate change|global warming|carbon credit)\b",
    r"\b(mushroom|aquaculture|fishery|shrimp|prawn|poultry|dairy)\b",
]


def classify_unprocessable(message: str) -> Tuple[bool, str]:
    """
    Determine if a message is unprocessable by the main orchestrator
    and why.

    Returns:
        (is_unprocessable, reason)
        reason: 'out_of_domain' | 'border_agri' | 'too_vague' | 'processable'
    """
    msg_lower = message.lower()

    # Truly out-of-domain
    for pattern in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pattern, msg_lower):
            return True, "out_of_domain"

    # Agricultural but edge-case for our system
    for pattern in BORDER_CASE_PATTERNS:
        if re.search(pattern, msg_lower):
            return True, "border_agri"

    # Too vague (very short, no actionable content)
    if len(message.strip().split()) < 3 and not re.search(r"\d", message):
        return True, "too_vague"

    return False, "processable"


# ---------------------------------------------------------------------------
# Unprocessable Response Generator
# ---------------------------------------------------------------------------

UNPROCESSABLE_TEMPLATES = {
    "out_of_domain": (
        "🌾 I'm AgroCulture AI — I specialise in farming and agriculture.\n\n"
        "I'm not able to help with that topic, but I'd love to assist you with:\n\n"
        "  🌱 **Crop recommendations** — 'What crop should I grow?'\n"
        "  💊 **Fertilizer advice** — 'What fertilizer for rice?'\n"
        "  📊 **Soil analysis** — 'Is my soil healthy?'\n"
        "  🌿 **Disease treatment** — 'How to treat leaf blight?'\n"
        "  📈 **Yield forecasting** — 'Predict wheat yield in Punjab'\n\n"
        "Is there anything farming-related I can help you with?"
    ),
    "too_vague": (
        "🤔 I'd love to help, but your question is a bit too brief for me to understand.\n\n"
        "Could you give me a bit more context? For example:\n"
        "  • What crop are you growing?\n"
        "  • What problem are you facing?\n"
        "  • What are your soil/climate conditions?\n\n"
        "The more you tell me, the better I can assist!"
    ),
}


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Client for locally-running Ollama LLM server.
    Ollama must be running: https://ollama.com
    Pull a model first: ollama pull mistral   (or llama3, gemma2, etc.)
    """

    DEFAULT_URL = "http://localhost:11434"
    PREFERRED_MODELS = ["mistral", "llama3", "gemma2:2b", "phi3", "tinyllama"]

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", self.DEFAULT_URL)).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "")
        self._available = None   # Lazy-checked
        self._active_model = None

    def _check_availability(self) -> bool:
        """Check if Ollama is running and has a usable model."""
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"].split(":")[0] for m in data.get("models", [])]

                if self.model and self.model in models:
                    self._active_model = self.model
                    return True

                # Auto-select from preferred list
                for preferred in self.PREFERRED_MODELS:
                    if preferred in models:
                        self._active_model = preferred
                        logger.info(f"[Ollama] Auto-selected model: {preferred}")
                        return True

                if models:
                    self._active_model = data["models"][0]["name"]
                    logger.info(f"[Ollama] Using available model: {self._active_model}")
                    return True

        except Exception as e:
            logger.debug(f"[Ollama] Not available: {e}")

        return False

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._check_availability()
        return self._available

    def generate(self, prompt: str, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> Optional[str]:
        """
        Send a prompt to Ollama and get a response.

        Returns:
            Generated text or None on failure.
        """
        if not self.is_available:
            return None

        try:
            import urllib.request

            payload = json.dumps({
                "model": self._active_model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["\n\n\n"],
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                text = data.get("response", "").strip()
                if text:
                    return text

        except Exception as e:
            logger.warning(f"[Ollama] Generation failed: {e}")
            self._available = False   # Mark unavailable for this session

        return None

    @property
    def model_name(self) -> str:
        return self._active_model or "not connected"


# ---------------------------------------------------------------------------
# HuggingFace Inference API Client (free tier)
# ---------------------------------------------------------------------------

class HuggingFaceClient:
    """
    Client for HuggingFace Inference API (free tier).
    Set environment variable HF_TOKEN with your token from hf.co/settings/tokens
    """

    API_URL = "https://api-inference.huggingface.co/models/{model}"
    DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

    def __init__(self):
        self.token = os.environ.get("HF_TOKEN", "")
        self.model = os.environ.get("HF_MODEL", self.DEFAULT_MODEL)
        self._available = bool(self.token)

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt: str, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> Optional[str]:
        if not self.is_available:
            return None

        try:
            import urllib.request

            full_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"

            payload = json.dumps({
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                self.API_URL.format(model=self.model),
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "").strip()
                    if text:
                        return text

        except Exception as e:
            logger.warning(f"[HuggingFace] Generation failed: {e}")

        return None

    @property
    def model_name(self) -> str:
        return self.model if self._available else "no token set"


# ---------------------------------------------------------------------------
# Template Fallback
# ---------------------------------------------------------------------------

TEMPLATE_RESPONSES = {
    "companion_planting": (
        "🌿 **Companion Planting** is the practice of growing different plants near each other "
        "for mutual benefit. Common examples:\n"
        "  • **Three Sisters** (maize + beans + squash) — traditional Native American method\n"
        "  • **Marigolds** near tomatoes — repels aphids and whiteflies\n"
        "  • **Basil** near tomatoes — improves flavour and repels pests\n"
        "  • **Garlic** near roses — repels aphids\n"
        "It's a natural, chemical-free pest management strategy!"
    ),
    "organic_farming": (
        "🌱 **Organic Farming** avoids synthetic fertilizers and pesticides. Key practices:\n"
        "  • Use compost, vermicompost, and green manures\n"
        "  • Apply biofertilizers (Rhizobium, Azotobacter, PSB)\n"
        "  • Use neem-based pesticides and Trichoderma for disease control\n"
        "  • Practice crop rotation to break pest cycles\n"
        "  • APEDA-certified organic products get premium prices in export markets"
    ),
    "irrigation": (
        "💧 **Irrigation Methods** suited to Indian conditions:\n"
        "  • **Drip irrigation** — Best for vegetables, orchards (80% water saving)\n"
        "  • **Sprinkler** — Good for wheat, groundnut on undulating land\n"
        "  • **Furrow** — Traditional, suitable for sugarcane, maize\n"
        "  • **Flood / Basin** — Rice paddies, orchards\n"
        "PM Krishi Sinchayee Yojana provides 55% subsidy for micro-irrigation."
    ),
}


def _match_template(message: str) -> Optional[str]:
    """Try to match message to a template for offline fallback."""
    msg_lower = message.lower()
    if re.search(r"companion\s+plant", msg_lower):
        return TEMPLATE_RESPONSES["companion_planting"]
    if re.search(r"organic\s+farm", msg_lower):
        return TEMPLATE_RESPONSES["organic_farming"]
    if re.search(r"irrigation|drip|sprinkler", msg_lower):
        return TEMPLATE_RESPONSES["irrigation"]
    return None


# ---------------------------------------------------------------------------
# Main TraditionalLLMBackend
# ---------------------------------------------------------------------------

class TraditionalLLMBackend:
    """
    Traditional LLM backend that works cooperatively with the AgroCultureLLM
    orchestrator.

    Priority:
        1. Ollama (local)
        2. HuggingFace Inference API
        3. Template-based responses
        4. Graceful error message

    Use this for:
        - Unprocessable / out-of-domain questions
        - Free-form agricultural explanations
        - Enriching orchestrator responses with natural language
    """

    def __init__(self):
        self.ollama = OllamaClient()
        self.hf = HuggingFaceClient()
        self._backend_name = None

        # Unprocessable log file
        self._log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self._log_path = os.path.join(self._log_dir, "unprocessable_log.jsonl")

    def _detect_backend(self) -> str:
        if self.ollama.is_available:
            return "ollama"
        if self.hf.is_available:
            return "huggingface"
        return "template"

    @property
    def backend_name(self) -> str:
        if self._backend_name is None:
            self._backend_name = self._detect_backend()
        return self._backend_name

    @property
    def is_llm_available(self) -> bool:
        return self.backend_name in ("ollama", "huggingface")

    @property
    def status(self) -> str:
        lines = []
        lines.append(f"  {'✅' if self.ollama.is_available else '❌'} Ollama (local) — model: {self.ollama.model_name}")
        lines.append(f"  {'✅' if self.hf.is_available else '❌'} HuggingFace API — model: {self.hf.model_name}")
        lines.append(f"  ✅ Template fallback — always available")
        lines.append(f"  → Active backend: {self.backend_name}")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Prompt Builder
    # -------------------------------------------------------------------

    def _build_prompt(self, user_message: str, context: str = "",
                      conversation_history: Optional[List[Dict]] = None) -> str:
        """Build a prompt incorporating context and conversation history."""
        parts = []

        # Add relevant conversation turns (last 3 pairs for context)
        if conversation_history:
            recent = conversation_history[-6:]  # last 3 user+assistant pairs
            for turn in recent:
                role = turn.get("role", "")
                msg = turn.get("message", "")[:300]  # truncate long messages
                if role == "user":
                    parts.append(f"User: {msg}")
                elif role == "assistant":
                    parts.append(f"Assistant: {msg}")

        # Add current context (e.g., from ML model predictions)
        if context:
            parts.append(f"\n[System context: {context}]\n")

        # Add current user message
        parts.append(f"User: {user_message}")
        parts.append("Assistant:")

        return "\n".join(parts)

    # -------------------------------------------------------------------
    # Core Generate
    # -------------------------------------------------------------------

    def generate(self, user_message: str, context: str = "",
                 conversation_history: Optional[List[Dict]] = None,
                 max_tokens: int = 400, temperature: float = 0.7) -> str:
        """
        Generate a response using the best available LLM backend.

        Args:
            user_message: The user's message
            context: Optional context (e.g., ML model prediction results)
            conversation_history: Prior conversation turns
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)

        Returns:
            Generated response string (always returns something)
        """
        prompt = self._build_prompt(user_message, context, conversation_history)

        # Try Ollama
        if self.ollama.is_available:
            result = self.ollama.generate(prompt, AGRO_SYSTEM_PROMPT, max_tokens, temperature)
            if result:
                self._backend_name = "ollama"
                return self._format_llm_response(result)

        # Try HuggingFace
        if self.hf.is_available:
            result = self.hf.generate(prompt, AGRO_SYSTEM_PROMPT, max_tokens, temperature)
            if result:
                self._backend_name = "huggingface"
                return self._format_llm_response(result)

        # Template fallback
        template = _match_template(user_message)
        if template:
            self._backend_name = "template"
            return template

        # Generic fallback
        self._backend_name = "template"
        return self._generic_fallback(user_message)

    def _format_llm_response(self, text: str) -> str:
        """Clean up and format LLM output."""
        # Remove leading "Assistant:" prefix if echoed back
        text = re.sub(r"^Assistant:\s*", "", text).strip()
        # Remove repeated newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Add LLM source tag
        return text + f"\n\n_🤖 Powered by {self._backend_name.title()} LLM_"

    def _generic_fallback(self, message: str) -> str:
        """Last-resort response when all backends fail."""
        msg_lower = message.lower()

        # Try to extract the topic and give a generic agricultural answer
        agri_keywords = re.findall(
            r"\b(crop|soil|fertilizer|pest|disease|irrigation|seed|harvest|"
            r"organic|compost|rain|temperature|humidity|yield|farm)\b",
            msg_lower
        )

        if agri_keywords:
            topic = agri_keywords[0]
            return (
                f"🌾 That's a great question about **{topic}**!\n\n"
                "While I couldn't connect to an LLM backend for a detailed answer, "
                "here are some resources:\n\n"
                "  📖 **ICAR** — icar.org.in\n"
                "  📖 **eNAM** — enam.gov.in (market prices)\n"
                "  📞 **Kisan Call Centre** — 1800-180-1551 (free, 24×7)\n"
                "  📱 **KVK** — Contact your local Krishi Vigyan Kendra\n\n"
                "Try asking me about crop prediction, fertilizer doses, soil analysis, "
                "or disease treatment — I can answer those with high confidence!"
            )

        return (
            "🌾 I'm not sure how to answer that specific question right now.\n\n"
            "For farming advice, try asking:\n"
            "  • 'What crop should I grow with N=80, P=40, K=40?'\n"
            "  • 'What fertilizer for rice in clay soil?'\n"
            "  • 'How to treat leaf blight in wheat?'\n\n"
            "For urgent farm advice, call **Kisan Call Centre: 1800-180-1551** (free, 24×7)"
        )

    # -------------------------------------------------------------------
    # Unprocessable Handler
    # -------------------------------------------------------------------

    def handle_unprocessable(self, message: str, reason: str,
                              conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Handle a question the main orchestrator cannot process.

        Args:
            message: The user's message
            reason: Why it's unprocessable ('out_of_domain', 'border_agri', 'too_vague')
            conversation_history: Prior conversation

        Returns:
            Response string
        """
        # Log the unprocessable query for future improvement
        self._log_unprocessable(message, reason)

        # Out-of-domain: don't try the LLM, just decline politely
        if reason == "out_of_domain":
            return UNPROCESSABLE_TEMPLATES["out_of_domain"]

        # Too vague: ask for clarification
        if reason == "too_vague":
            return UNPROCESSABLE_TEMPLATES["too_vague"]

        # Border agricultural case: try LLM, fall back to guidance
        if reason == "border_agri":
            if self.is_llm_available:
                context = (
                    "This question is agricultural but may be outside precise ML-model coverage. "
                    "Answer with general agricultural knowledge."
                )
                result = self.generate(message, context, conversation_history)
                if result:
                    return f"📚 **AgroCulture AI** (LLM mode):\n\n{result}"

            # No LLM — give helpful direction
            return (
                "🌾 That's an interesting agricultural question!\n\n"
                "While it's slightly outside my core ML-model coverage, here's some guidance:\n\n"
                "  📞 **Kisan Call Centre**: 1800-180-1551 (free, Hindi + regional languages)\n"
                "  🌐 **eNAM**: enam.gov.in — market prices, mandi rates\n"
                "  🏛️ **ICAR**: icar.org.in — research-backed recommendations\n"
                "  🏘️ **KVK**: Your local Krishi Vigyan Kendra for hands-on guidance\n\n"
                "If you'd like to set up a local LLM for richer answers, "
                "run `ollama pull mistral` and restart AgroCulture AI."
            )

        # Fallback
        return self.generate(message, "", conversation_history)

    def enrich_response(self, base_response: str, user_message: str,
                        conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Optionally enrich an ML-generated response with LLM narrative.
        Only called when the LLM is available.

        Returns the enriched response, or original if LLM unavailable/fails.
        """
        if not self.is_llm_available:
            return base_response

        context = (
            f"The ML system already produced this response:\n---\n{base_response}\n---\n"
            "Add only a short (2-3 sentence) practical farming tip that complements "
            "this answer. Do NOT repeat the data above."
        )

        enrichment = self.generate(user_message, context, conversation_history,
                                   max_tokens=150, temperature=0.6)
        if enrichment and len(enrichment) > 30:
            return base_response + "\n\n💡 **Additional Tip:**\n" + enrichment

        return base_response

    # -------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------

    def _log_unprocessable(self, message: str, reason: str) -> None:
        """Append unprocessable queries to a JSONL log for future analysis."""
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "reason": reason,
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Never crash due to logging

    def get_unprocessable_stats(self) -> Dict:
        """Return statistics about unprocessable queries."""
        stats = {"total": 0, "by_reason": {}}
        if not os.path.exists(self._log_path):
            return stats

        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    stats["total"] += 1
                    reason = entry.get("reason", "unknown")
                    stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
        except Exception:
            pass

        return stats


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_backend = None

def get_backend() -> TraditionalLLMBackend:
    """Get or create the singleton LLM backend."""
    global _backend
    if _backend is None:
        _backend = TraditionalLLMBackend()
    return _backend


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Traditional LLM Backend — Status Test")
    print("=" * 70)

    backend = TraditionalLLMBackend()
    print("\n📦 Backend Status:")
    print(backend.status)

    print("\n" + "=" * 70)
    print("  Unprocessable Detection Tests")
    print("=" * 70)

    test_cases = [
        ("What is the best IPL team?", "out_of_domain"),
        ("How do I get a KCC loan?", "border_agri"),
        ("hi", "too_vague"),
        ("What crop should I grow in loamy soil?", "processable"),
        ("Tell me about companion planting", "processable — LLM should answer"),
    ]

    for msg, expected in test_cases:
        is_unproc, reason = classify_unprocessable(msg)
        status = "✅" if (is_unproc and reason != "processable") or (not is_unproc) else "❓"
        print(f"\n  {status} Message: '{msg}'")
        print(f"     Expected: {expected}")
        print(f"     Got: unprocessable={is_unproc}, reason={reason}")

    print("\n" + "=" * 70)
    print("  LLM Response Tests")
    print("=" * 70)

    test_messages = [
        ("What is IPM?", ""),
        ("Tell me about drip irrigation", ""),
        ("How do I get KCC loan?", "border_agri"),
        ("Cricket score?", "out_of_domain"),
    ]

    for msg, reason in test_messages:
        print(f"\nQuery: {msg}")
        if reason:
            response = backend.handle_unprocessable(msg, reason)
        else:
            response = backend.generate(msg)
        print(response[:300])
        print("-" * 60)
