# 🌾 AgroCulture AI — Custom Agriculture LLM System

> An intelligent, conversational AI system for Indian agriculture that combines **3 pre-trained ML models**, a **RAG knowledge base** (3,870 documents), and a **rule-based fertilizer advisor** into a unified natural-language chat interface.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [LLM Workflow — Step by Step](#llm-workflow--step-by-step)
4. [Connected Models](#connected-models)
5. [Knowledge Base (RAG Pipeline)](#knowledge-base-rag-pipeline)
6. [Intent Classification Engine](#intent-classification-engine)
7. [Fertilizer Rule Engine](#fertilizer-rule-engine)
8. [Data Flow for Each Query Type](#data-flow-for-each-query-type)
9. [File Structure & Module Map](#file-structure--module-map)
10. [Datasets Used](#datasets-used)
11. [How to Run](#how-to-run)
12. [Technical Details](#technical-details)

---

## System Overview

AgroCulture AI is **not** a traditional transformer-based LLM. It is a **hybrid AI orchestrator** that intelligently routes natural-language farming questions to the right combination of:

| Component | Technology | Role |
|---|---|---|
| **Intent Classifier** | Regex + keyword NLP | Understands _what_ the user is asking |
| **Entity Extractor** | Regex pattern matching | Pulls out N, P, K, pH, crop, soil type, etc. |
| **Crop Prediction Model** | scikit-learn RandomForest (.pkl) | Predicts the best crop for given conditions |
| **Fertilizer Prediction Model** | scikit-learn classifier (.pkl) | Predicts fertilizer name from soil/crop data |
| **Yield Forecast Model** | scikit-learn GradientBoosting (.pkl) | Predicts crop production for a state/year |
| **Fertilizer Rule Engine** | Python rule-based calculator | Computes NPK deficits → fertilizer dosage + cost |
| **RAG Knowledge Base** | sentence-transformers + FAISS | Searches 3,870 agricultural knowledge documents |
| **Response Generator** | Template-based NLG | Formats predictions into human-readable answers |

The system answers **6 types of agricultural queries**:
- 🌱 Crop prediction
- 💊 Fertilizer recommendation (name + dosage + cost)
- 📊 Soil health analysis
- 🌿 Crop disease remediation
- 📈 State-wise yield forecasting
- 📚 General agricultural Q&A

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT                                     │
│  "What crop should I grow with N=80, P=40, K=40, pH 6.5, temp 20°C,   │
│   humidity 50%, rainfall 75mm in loamy soil?"                           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    INTENT CLASSIFIER (intent_engine.py)                  │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────────────────────────────┐   │
│  │ Keyword Matching  │    │ Regex Pattern Matching                    │   │
│  │ "what crop" → 1.0 │    │ r"(?:what|which)\s+crop" → 1.5          │   │
│  │ "suggest" → 1.0   │    │ Confidence = sum / 5.0                   │   │
│  └──────────────────┘    └──────────────────────────────────────────┘   │
│                                                                         │
│  Result: intent = "crop_prediction", confidence = 0.50                  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENTITY EXTRACTOR (intent_engine.py)                   │
│                                                                         │
│  Extracted from natural language:                                        │
│    nitrogen: 80.0    (matched "N=80" — case-sensitive regex)            │
│    phosphorus: 40.0  (matched "P=40")                                   │
│    potassium: 40.0   (matched "K=40")                                   │
│    ph: 6.5           (matched "pH 6.5")                                 │
│    temperature: 20.0 (matched "20°C")                                   │
│    humidity: 50.0    (matched "50%")                                    │
│    rainfall: 75.0    (matched "75mm")                                   │
│    soil_type: Loamy  (word-boundary match from SOIL_TYPES list)         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MISSING FIELD CHECK                                   │
│                                                                         │
│  Required for crop_prediction:                                          │
│    [✓] nitrogen  [✓] phosphorus  [✓] potassium  [✓] temperature        │
│    [✓] humidity  [✓] ph          [✓] rainfall   [✓] soil_type          │
│                                                                         │
│  All fields present → proceed to model inference                        │
│  (If fields were missing → return prompt asking for them)               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌────────────────┐ ┌───────────────────┐
│   ML MODEL       │ │ RULE ENGINE    │ │ RAG KNOWLEDGE     │
│ (crop_pred.pkl)  │ │ (fert_advisor) │ │ BASE (FAISS)      │
│                  │ │                │ │                   │
│ Input DataFrame: │ │ Calculates NPK │ │ Searches for      │
│ N=80, P=40, K=40 │ │ deficits for   │ │ "Tomato cultiv-   │
│ Temp=20, Hum=50  │ │ predicted crop │ │ ation require-    │
│ pH=6.5, Rain=75  │ │ vs current     │ │ ments" across     │
│ Soil=Loamy       │ │ soil levels    │ │ 3,870 documents   │
│                  │ │                │ │                   │
│ Output:          │ │ Output:        │ │ Output:           │
│ crop = "Tomato"  │ │ DAP 65 kg/ha   │ │ Top-3 relevant    │
│ (via label       │ │ Urea 61 kg/ha  │ │ context docs      │
│  encoder)        │ │ MOP 67 kg/ha   │ │                   │
└────────┬─────────┘ └───────┬────────┘ └─────────┬─────────┘
         └──────────────────┬┘                    │
                            ▼                     │
┌─────────────────────────────────────────────────┴───────────────────────┐
│                    RESPONSE GENERATOR                                   │
│                                                                         │
│  🌱 **Crop Prediction Result**                                          |
│  Based on your soil and climate conditions:                             │
│    • Nitrogen: 80.0  • Phosphorus: 40.0  • Potassium: 40.0              │
│    • Temperature: 20.0°C  • pH: 6.5  • Rainfall: 75.0 mm                │
│                                                                         │
│  🎯 **Recommended Crop: Tomato**                                        │
│                                                                          │
│  💊 **Fertilizer Suggestion for Tomato:**                               │
│    • DAP: 65.2 kg/ha — Phosphorus deficit of 30.0 kg/ha                 │
│    • Urea: 61.4 kg/ha — Nitrogen deficit of 28.3 kg/ha                  │
│    • MOP: 66.7 kg/ha — Potassium deficit of 40.0 kg/ha                  │
│    💰 Est. Cost: ₹3,263/hectare                                         │
│                                                                         │
│  📚 **Additional Information:**                                         │
│    • Tomato in Loamy soil: Ideal N=97, P=69, K=47, Temp=25.1°C...       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Workflow — Step by Step

### Phase 1: User Input Processing

```
User types: "What fertilizer for sugarcane? My soil has N=40, P=10, K=80"
```

1. **`chat.py`** receives the raw text input from the terminal
2. Passes it to **`AgroCultureLLM.chat(message)`** in `agro_llm.py`
3. Message is stored in `conversation_history` for multi-turn context

### Phase 2: Intent Classification

```python
# intent_engine.py → IntentClassifier.classify()
intent, confidence = classifier.classify(message)
# Result: ("fertilizer_recommendation", 1.0)
```

The classifier scores each intent by:
- **Keyword matching** (+1.0 per keyword hit): `"fertilizer"` matches → +1.0
- **Regex pattern matching** (+1.5 per pattern hit): `r"(?:what|which)\s+fertilizer"` matches → +1.5
- **Confidence** = `min(total_score / 5.0, 1.0)`

Supported intents:

| Intent | Trigger Keywords | Trigger Patterns |
|---|---|---|
| `crop_prediction` | "what crop", "suggest crop", "best crop" | `r"(?:what\|which)\s+(?:crop\|plant)"` |
| `fertilizer_recommendation` | "fertilizer", "NPK", "nutrient", "urea" | `r"(?:what\|which)\s+fertili[sz]er"` |
| `soil_analysis` | "soil health", "soil test", "analyze soil" | `r"(?:soil\|land)\s+(?:health\|fertility)"` |
| `disease_query` | "disease", "blight", "rot", "remedy" | `r"(?:disease\|pest\|infection)"` |
| `yield_forecast` | "yield", "production", "forecast" | `r"(?:predict\|forecast)\s+.*(?:yield\|production)"` |
| `general_qa` | _(fallback when no intent matches)_ | — |

### Phase 3: Entity Extraction

```python
# intent_engine.py → IntentClassifier.extract_entities()
entities = classifier.extract_entities(message)
# Result: {nitrogen: 40.0, phosphorus: 10.0, potassium: 80.0, crop: "Sugarcane"}
```

Extraction uses **two-pass regex**:
- **Case-sensitive pass** for N, P, K (uppercase chemical symbols): `N=40` → nitrogen=40.0
- **Case-insensitive pass** for spelled-out names: `temperature`, `humidity`, `rainfall`, `ph`
- **Word-boundary matching** for categorical values: soil types, crop names, Indian states, seasons

### Phase 4: Missing Field Validation

```python
# Check what's required vs what's extracted
missing = classifier.get_missing_fields("fertilizer_recommendation", entities)
# If critical fields are missing → return prompt asking for them
# If enough fields → proceed to model inference
```

Each intent has a required field list:

| Intent | Required Fields |
|---|---|
| `crop_prediction` | N, P, K, temperature, humidity, pH, rainfall, soil_type |
| `fertilizer_recommendation` | temperature, humidity, moisture, soil_type, crop, N, K, P |
| `soil_analysis` | N, P, K, pH |
| `yield_forecast` | state, crop |
| `disease_query` | _(none — uses free-text search)_ |

### Phase 5: Intent Routing & Model Inference

Based on the classified intent, `agro_llm.py` routes to the appropriate handler:

```python
if intent == "crop_prediction":
    response = self._handle_crop_prediction(entities, missing)
elif intent == "fertilizer_recommendation":
    response = self._handle_fertilizer_recommendation(entities, missing)
elif intent == "soil_analysis":
    response = self._handle_soil_analysis(entities)
elif intent == "disease_query":
    response = self._handle_disease_query(message)
elif intent == "yield_forecast":
    response = self._handle_yield_forecast(entities, missing)
else:
    response = self._handle_general_qa(message)
```

### Phase 6: Response Assembly

Each handler combines results from multiple sources:
1. **ML model prediction** (primary result)
2. **Rule engine output** (fertilizer dosage, soil classification)
3. **RAG knowledge retrieval** (contextual enrichment)
4. **Template formatting** (emojis, sections, tables)

### Phase 7: Multi-Turn Context

Entities accumulate across conversation turns:
```
Turn 1: "I want to grow rice"         → entities: {crop: "Rice"}
Turn 2: "My soil has N=90, P=50, K=40" → entities: {crop: "Rice", N: 90, P: 50, K: 40}
Turn 3: "What fertilizer?"            → uses accumulated entities for recommendation
```

The `/reset` command clears accumulated entities and conversation history.

---

## Connected Models

### Model 1: Crop Prediction Model

```
Location:  model/crop_model/crop_prediction_model.pkl
Encoder:   model/crop_model/crop_label_encoder.pkl
Type:      scikit-learn RandomForestClassifier
Training:  sensor_Crop_Dataset (1).csv — 20,000 rows
```

**Input Features (9):**

| Feature | Type | Example |
|---|---|---|
| Nitrogen | float | 80.0 |
| Phosphorus | float | 40.0 |
| Potassium | float | 40.0 |
| Temperature | float | 20.0 |
| Humidity | float | 50.0 |
| pH_Value | float | 6.5 |
| Rainfall | float | 75.0 |
| Soil_Type | categorical | Loamy |
| Variety | categorical | (empty string) |

**Output:** Encoded crop label → decoded via `crop_label_encoder.pkl` → crop name (e.g., "Tomato", "Wheat", "Rice")

**Integration in agro_llm.py:**
```python
# Line 42-43
CROP_MODEL_PATH = model/crop_model/crop_prediction_model.pkl
CROP_ENCODER_PATH = model/crop_model/crop_label_encoder.pkl

# _handle_crop_prediction():
input_data = pd.DataFrame([{N, P, K, Temp, Humidity, pH, Rainfall, Soil_Type, Variety}])
prediction = self._crop_model.predict(input_data)
crop_name = self._crop_encoder.inverse_transform(prediction)[0]
```

**What happens after prediction:**
- The predicted crop is passed to the **Fertilizer Rule Engine** to auto-calculate NPK deficits
- The predicted crop is searched in the **RAG Knowledge Base** for additional cultivation info

---

### Model 2: Fertilizer Prediction Model

```
Location:  model/fartilizer_model/fertilizer_prediction_model.pkl
Encoder:   model/fartilizer_model/fertilizer_label_encoder.pkl
Type:      scikit-learn classifier
Training:  f2.csv — 553 rows
```

**Input Features (8):**

| Feature | Type | Example |
|---|---|---|
| Temparature | float | 25.0 |
| Humidity | float | 84.0 |
| Moisture | float | 32.0 |
| Soil_Type | categorical | Clayey |
| Crop_Type | categorical | Rice |
| Nitrogen | float | 90.0 |
| Potassium | float | 49.0 |
| Phosphorous | float | 36.0 |

**Output:** Fertilizer name (e.g., "Urea", "DAP", "14-35-14", "28-28", "10-26-26")

**Integration in agro_llm.py:**
```python
# Line 44-45
FERT_MODEL_PATH = model/fartilizer_model/fertilizer_prediction_model.pkl
FERT_ENCODER_PATH = model/fartilizer_model/fertilizer_label_encoder.pkl

# _predict_fertilizer_ml():
input_data = pd.DataFrame([{Temp, Humidity, Moisture, Soil, Crop, N, K, P}])
prediction = self._fert_model.predict(input_data)
fert_name = self._fert_encoder.inverse_transform(prediction)[0]
```

**Dual recommendation system:**
The ML model prediction (fertilizer _name_) is **combined** with the rule engine (fertilizer _dosage_). The user gets both:
- 🤖 ML prediction: "Urea"
- 📊 Rule engine: "Urea 192.3 kg/ha (₹1,154/ha) + DAP 119.6 kg/ha (₹3,228/ha)"

---

### Model 3: State Crop Yield / Production Forecast Model

```
Location:  model/state_crop_yeild/models/production_model.pkl
Metadata:  model/state_crop_yeild/models/model_metadata.json
Type:      scikit-learn Pipeline (OneHotEncoder + RandomForestRegressor or GradientBoostingRegressor)
Training:  state_wise_crop_yild.csv — 19,690 rows
```

**Input Features (8):**

| Feature | Type | Example |
|---|---|---|
| State | categorical | Punjab |
| Crop | categorical | Wheat |
| Season | categorical | Rabi |
| Crop_Year | int | 2026 |
| Area | float | 3500.0 (hectares) |
| Annual_Rainfall | float | 636.5 (mm) |
| Fertilizer | float | 766879.0 |
| Pesticide | float | 2497.0 |

**Output:** Predicted Production (tonnes) → Estimated Yield = Production / Area

**Smart defaults:** When the user provides only state + crop + year, the system auto-fills Area, Rainfall, Fertilizer, and Pesticide from **historical averages** (last 3 years of available data in `state_wise_crop_yild.csv`).

**Integration in agro_llm.py:**
```python
# Line 46-48
YIELD_MODEL_PATH = model/state_crop_yeild/models/production_model.pkl
YIELD_META_PATH  = model/state_crop_yeild/models/model_metadata.json
YIELD_DATASET    = dataset/state_wise_crop_yild.csv

# _handle_yield_forecast():
defaults = get_historical_defaults(dataset, state, crop)  # Last 3 years avg
input_df = pd.DataFrame([{State, Crop, Season, Year, Area, Rainfall, Fertilizer, Pesticide}])
production = max(0, self._yield_model.predict(input_df)[0])
yield_per_ha = production / area
```

---

## Knowledge Base (RAG Pipeline)

### How It's Built (`build_knowledge_base.py`)

```
8 CSV Datasets  →  Document Generators  →  3,870 Text Chunks
                                                    │
                                           sentence-transformers
                                           (all-MiniLM-L6-v2)
                                                    │
                                              384-dim Vectors
                                                    │
                                              FAISS IndexFlatIP
                                                    │
                                         model/llm/embeddings/
                                         model/llm/data/
```

### Document Sources

| Dataset | Rows | Documents Generated | Category |
|---|---|---|---|
| `sensor_Crop_Dataset (1).csv` | 20,000 | 36 (grouped by Crop×Soil) | `crop_requirements` |
| `Fertilizer Prediction.csv` | 99 | 64 (grouped by Crop×Soil×Fertilizer) | `fertilizer_recommendation` |
| `f2.csv` | 552 | 84 (grouped by Crop×Soil×Fertilizer) | `fertilizer_recommendation` |
| `soil_fertility_dataset.csv` | 880 | 881 (1 per sample + 1 summary) | `soil_analysis` |
| `Soil-Climate-data.csv` | 10,000 | 76 (grouped by Crop×Soil) | `crop_soil_compatibility` |
| `merged_rainfall_dataset.csv` | 733 | 729 (1 per district) | `rainfall_info` |
| `crop_disease_remediation.csv` | 2,000* | 2,000 (1 per Q&A pair) | `disease_info` |
| **Total** | | **3,870 documents** | |

_*Limited to 2,000 rows from the 122MB dataset for index size management._

### How Search Works

```python
# User asks: "How to treat white rot in onion?"
query_embedding = model.encode(["How to treat white rot in onion?"])  # 384-dim vector
scores, indices = faiss_index.search(query_embedding, top_k=5)       # Cosine similarity
# Returns top-5 most similar documents from 3,870 indexed chunks
```

### Example Search Results

| Query | Top Result | Score |
|---|---|---|
| "Fertilizer for rice in loamy soil" | "For rice in Loamy soil... recommended fertilizer is Urea" | 0.79 |
| "Rainfall in Punjab" | "Average annual rainfall in Ludhiana, Punjab is 684.3 mm" | 0.85 |
| "White rot disease" | "Crop: Onion, Disease: White Rot, Type: Fungal..." | 0.57 |
| "Sugarcane cultivation" | "Sugarcane in Sandy soil: Ideal N=76, P=46, K=55..." | 0.79 |

---

## Intent Classification Engine

### Scoring Algorithm

```
For each intent:
    score = 0
    for keyword in intent.keywords:
        if keyword in message.lower():
            score += 1.0
    for pattern in intent.regex_patterns:
        if re.search(pattern, message.lower()):
            score += 1.5

    confidence = min(score / 5.0, 1.0)

best_intent = intent with highest score
```

### Entity Extraction — Two-Pass Strategy

**Pass 1 — Case-Sensitive (for N, P, K symbols):**
```
"N=80"  → regex: (?:nitrogen|N)\s*[=:]\s*(\d+\.?\d*)  → nitrogen: 80.0
"P=40"  → regex: (?:phosphorus|P)\s*[=:]\s*(\d+\.?\d*) → phosphorus: 40.0
"K=40"  → regex: (?:potassium|K)\s*[=:]\s*(\d+\.?\d*)  → potassium: 40.0
```

**Pass 2 — Case-Insensitive (for everything else):**
```
"temperature 20°C" → regex: (\d+\.?\d*)\s*°?\s*[cC]     → temperature: 20.0
"humidity 50%"     → regex: humidity\s+(?:is\s+)?(\d+)   → humidity: 50.0
"pH 6.5"           → regex: (?:ph|pH)\s*[=:]\s*(\d+)     → ph: 6.5
"75mm"             → regex: (\d+\.?\d*)\s*mm              → rainfall: 75.0
"loamy"            → word-boundary match in SOIL_TYPES     → soil_type: Loamy
"sugarcane"        → word-boundary match in CROP_NAMES     → crop: Sugarcane
"punjab"           → substring match in INDIAN_STATES      → state: Punjab
```

---

## Fertilizer Rule Engine

### How NPK Deficit Calculation Works

```
Step 1: Look up ideal NPK range for target crop
        Sugarcane → N: (120, 180), P: (50, 80), K: (60, 100)

Step 2: Calculate midpoint of ideal range
        N ideal = 150, P ideal = 65, K ideal = 80

Step 3: Compute deficit = max(0, ideal_mid - current_soil)
        Soil: N=40, P=10, K=80
        N deficit = 150 - 40 = 110 kg/ha
        P deficit = 65 - 10 = 55 kg/ha
        K deficit = 80 - 80 = 0 kg/ha  (sufficient!)

Step 4: Map deficits to commercial fertilizers
        P deficit → DAP (46% P₂O₅):  55 / 0.46 = 119.6 kg/ha
                     DAP also provides N: 119.6 × 0.18 = 21.5 kg N
        Remaining N → Urea (46% N): (110 - 21.5) / 0.46 = 192.3 kg/ha
        K deficit = 0 → No MOP needed

Step 5: Calculate cost
        DAP: 119.6 kg × ₹27/kg = ₹3,228/ha
        Urea: 192.3 kg × ₹6/kg = ₹1,154/ha
        Total: ₹4,382/hectare
```

### Fertilizer Database

| Fertilizer | N% | P% | K% | Cost (₹/kg) | Use Case |
|---|---|---|---|---|---|
| Urea | 46 | 0 | 0 | 6 | Primary nitrogen source |
| DAP | 18 | 46 | 0 | 27 | Phosphorus + some nitrogen |
| SSP | 0 | 16 | 0 | 9 | Budget phosphorus option |
| MOP | 0 | 0 | 60 | 17 | Primary potassium source |
| NPK 10-26-26 | 10 | 26 | 26 | 25 | Balanced P+K |
| NPK 14-35-14 | 14 | 35 | 14 | 26 | High-P for flowering |
| NPK 28-28-0 | 28 | 28 | 0 | 24 | Balanced N+P |
| Ammonium Sulphate | 21 | 0 | 0 | 10 | N + Sulphur deficiency |

### Crop NPK Ideal Ranges

The system has ideal NPK ranges for **40+ Indian crops** including: Rice, Wheat, Maize, Sugarcane, Cotton, Tea, Coffee, Potato, Tomato, Onion, Groundnut, Soybean, Mustard, Banana, Mango, Grape, Chilli, Turmeric, Ginger, Barley, Peas, Lentil, Chickpea, Bajra, Jowar, Ragi, Arhar, Rubber, Pepper, Cardamom, Cabbage, Cauliflower, Brinjal, Watermelon, Papaya, and more.

---

## Data Flow for Each Query Type

### 1. Crop Prediction Flow

```
User: "What crop for N=80, P=40, K=40, temp 20, pH 6.5, rainfall 75, loamy?"
  │
  ├→ Intent: crop_prediction (confidence: 0.50)
  ├→ Entities: {N:80, P:40, K:40, temp:20, hum:50, pH:6.5, rain:75, soil:Loamy}
  │
  ├→ [ML Model] crop_prediction_model.pkl → predict(DataFrame) → "Tomato"
  ├→ [Rule Engine] fertilizer_advisor → deficit calc for Tomato → DAP+Urea+MOP
  ├→ [RAG KB] search("Tomato cultivation") → top-2 context documents
  │
  └→ Response: Crop=Tomato + Fertilizer doses + Cultivation tips
```

### 2. Fertilizer Recommendation Flow

```
User: "What fertilizer for sugarcane? N=40, P=10, K=80"
  │
  ├→ Intent: fertilizer_recommendation (confidence: 1.0)
  ├→ Entities: {crop:Sugarcane, N:40, P:10, K:80}
  │
  ├→ [Rule Engine] NPK deficit → DAP 119.6 kg/ha + Urea 192.3 kg/ha
  ├→ [ML Model] fertilizer_prediction_model.pkl → "Urea"
  ├→ [RAG KB] search("fertilizer for sugarcane") → related recommendations
  │
  └→ Response: Deficit analysis + Dosage + Cost + ML prediction + KB context
```

### 3. Soil Analysis Flow

```
User: "Analyze my soil: N=138, P=8.6, K=560, pH=7.46"
  │
  ├→ Intent: soil_analysis (confidence: 1.0)
  ├→ Entities: {N:138, P:8.6, K:560, pH:7.46}
  │
  ├→ [Rule Engine] classify: N=High, P=Low, K=High, pH=Neutral
  ├→ [RAG KB] search("soil N=138 P=8.6") → similar soil profiles
  │
  └→ Response: N🟢 P🔴 K🟢 pH🟢 + "Low phosphorus — apply DAP" + similar profiles
```

### 4. Disease Query Flow

```
User: "How to treat white rot in onion?"
  │
  ├→ Intent: disease_query (confidence: 1.0)
  ├→ Entities: {crop: Onion}
  │
  ├→ [RAG KB] FAISS search → top-5 disease documents with scores
  │  Score 0.71: "Onion White Rot — Fungal — Sclerotium cepivorum..."
  │  Score 0.70: "White Rot treatment — organic/IPM remedies..."
  │
  └→ Response: Disease info + Causal agent + Severity + Season + Remedies
```

### 5. Yield Forecast Flow

```
User: "Predict wheat production in Punjab for 2026"
  │
  ├→ Intent: yield_forecast (confidence: 0.8)
  ├→ Entities: {crop:Wheat, state:Punjab, year:2026}
  │
  ├→ [Historical Defaults] state_wise_crop_yild.csv → last 3 years avg
  │   Area=3500, Rainfall=636, Fertilizer=766879, Pesticide=2497
  │
  ├→ [ML Model] production_model.pkl → predict(DataFrame) → 4,523,100 tonnes
  │
  └→ Response: Production forecast + Yield/hectare + Historical data used
```

### 6. General Q&A Flow

```
User: "Tell me about rice cultivation"
  │
  ├→ Intent: general_qa (confidence: 0.0 — no specific intent matched)
  │
  ├→ [RAG KB] FAISS search → top-5 documents
  │   Score 0.72: "Rice in Clay soil: N=87, P=49, Temp=24.8°C..."
  │   Score 0.72: "Rice in Loamy soil: N=85, P=49, Temp=24.4°C..."
  │
  └→ Response: Knowledge base results with sources and relevance scores
```

---

## File Structure & Module Map

```
code/
├── chat.py                                    ← ENTRY POINT: python chat.py
│   └── Imports: agro_llm.py
│
├── requirements.txt                           ← Dependencies
│
├── dataset/                                   ← Raw training data (8 CSVs)
│   ├── sensor_Crop_Dataset (1).csv            (20,000 rows — crop features)
│   ├── Fertilizer Prediction.csv              (99 rows — fertilizer labels)
│   ├── f2.csv                                 (552 rows — fertilizer data)
│   ├── soil_fertility_dataset.csv             (880 rows — soil N/P/K/pH/OC)
│   ├── Soil-Climate-data.csv                  (10,000 rows — soil-climate compat)
│   ├── merged_rainfall_dataset.csv            (733 rows — district rainfall)
│   ├── crop_disease_remediation.csv           (122MB — disease Q&A)
│   └── state_wise_crop_yild.csv               (19,690 rows — state production)
│
├── model/
│   ├── crop_model/                            ← CROP PREDICTION
│   │   ├── crop_prediction_model.pkl          (2.9 MB — trained RandomForest)
│   │   ├── crop_label_encoder.pkl             (label decoder)
│   │   ├── trainClassifier.ipynb              (training notebook)
│   │   └── test_predict.ipynb                 (testing notebook)
│   │
│   ├── fartilizer_model/                      ← FERTILIZER PREDICTION
│   │   ├── fertilizer_prediction_model.pkl    (492 KB — trained classifier)
│   │   ├── fertilizer_label_encoder.pkl       (label decoder)
│   │   ├── fartilizer_training_classifier.ipynb
│   │   └── predict_fertilizer.ipynb
│   │
│   ├── state_crop_yeild/                      ← YIELD FORECASTING
│   │   ├── train.py                           (training script)
│   │   ├── test_predict.py                    (prediction + forecast CLI)
│   │   ├── models/
│   │   │   ├── production_model.pkl           (140 MB — trained pipeline)
│   │   │   └── model_metadata.json            (metrics, feature info)
│   │   └── state_crop_yield_models/           (per-state models)
│   │
│   ├── llm/                                   ← LLM ORCHESTRATION LAYER
│   │   ├── agro_llm.py                        ★ Core orchestrator (750 lines)
│   │   │   └── AgroCultureLLM class
│   │   │       ├── chat(message) → response
│   │   │       ├── _handle_crop_prediction()
│   │   │       ├── _handle_fertilizer_recommendation()
│   │   │       ├── _handle_soil_analysis()
│   │   │       ├── _handle_disease_query()
│   │   │       ├── _handle_yield_forecast()
│   │   │       └── _handle_general_qa()
│   │   │
│   │   ├── intent_engine.py                   ★ NLP layer (420 lines)
│   │   │   └── IntentClassifier class
│   │   │       ├── classify(message) → (intent, confidence)
│   │   │       ├── extract_entities(message) → {key: value}
│   │   │       ├── get_missing_fields(intent, entities) → [fields]
│   │   │       └── format_missing_prompt(missing) → prompt
│   │   │
│   │   ├── fertilizer_advisor.py              ★ Rule engine (450 lines)
│   │   │   └── FertilizerAdvisor class
│   │   │       ├── analyze_soil(data) → classification
│   │   │       ├── calculate_deficit(soil, crop) → NPK deficits
│   │   │       ├── recommend_fertilizers(deficits) → dosages
│   │   │       └── format_recommendation(result) → text
│   │   │
│   │   ├── build_knowledge_base.py            ★ RAG builder (590 lines)
│   │   │   ├── generate_crop_docs()
│   │   │   ├── generate_fertilizer_docs()
│   │   │   ├── generate_soil_fertility_docs()
│   │   │   ├── generate_soil_climate_docs()
│   │   │   ├── generate_rainfall_docs()
│   │   │   ├── generate_disease_docs()
│   │   │   ├── build_index() — FAISS indexing
│   │   │   └── KnowledgeBase class
│   │   │       ├── load() → bool
│   │   │       └── search(query, top_k) → results
│   │   │
│   │   ├── data/
│   │   │   └── knowledge_chunks.json          (3,870 text documents)
│   │   │
│   │   └── embeddings/
│   │       └── faiss_index/
│   │           └── index.faiss                (384-dim vector index)
│   │
│   ├── nlp/
│   │   └── input_pipeline.ipynb               (early-stage NLP experiments)
│   │
│   └── rainfall_models/
│       └── train_rainfall.ipynb
│
└── mcp data/
    └── README.md                              ← THIS FILE
```

---

## Datasets Used

### Primary Training Data

| Dataset | Size | Used By | Key Columns |
|---|---|---|---|
| `sensor_Crop_Dataset (1).csv` | 20,000 rows | Crop model + RAG KB | N, P, K, Temp, Humidity, pH, Rainfall, Crop, Soil_Type, Variety |
| `f2.csv` | 552 rows | Fertilizer model + RAG KB | Temp, Humidity, Moisture, Soil, Crop, N, K, P, Fertilizer |
| `state_wise_crop_yild.csv` | 19,690 rows | Yield model | State, Crop, Season, Year, Area, Production, Rainfall, Fertilizer, Pesticide, Yield |

### Knowledge Base Sources

| Dataset | Size | Documents | Content |
|---|---|---|---|
| `Fertilizer Prediction.csv` | 99 rows | 64 docs | Crop-soil-fertilizer mappings |
| `soil_fertility_dataset.csv` | 880 rows | 881 docs | Soil NPK/pH/OC → fertile/not fertile |
| `Soil-Climate-data.csv` | 10,000 rows | 76 docs | Crop-soil compatibility rates |
| `merged_rainfall_dataset.csv` | 733 rows | 729 docs | District-level rainfall + crop suitability |
| `crop_disease_remediation.csv` | 122 MB | 2,000 docs | Disease Q&A, remedies, severity |

---

## How to Run

### Prerequisites

```bash
# Install dependencies (from project root)
pip install -r requirements.txt
```

### Step 1: Build Knowledge Base (one-time)

```bash
python model/llm/build_knowledge_base.py --test
```

This downloads the `all-MiniLM-L6-v2` model (~80MB), processes all datasets, generates 3,870 documents, and creates the FAISS index. Takes ~1-2 minutes.

### Step 2: Start Chat

```bash
# Full system (with RAG)
python chat.py

# Fast startup (skip RAG — ML models only)
python chat.py --no-kb

# Run automated tests
python chat.py --test
```

### Chat Commands

| Command | Action |
|---|---|
| `/help` | Show usage examples |
| `/models` | Show model loading status |
| `/reset` | Clear conversation context |
| `/quit` | Exit |

---

## Technical Details

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | 1.8+ | ML model inference |
| pandas | 2.0+ | Data processing |
| numpy | 2.0+ | Numerical operations |
| sentence-transformers | 5.6+ | Text embedding for RAG |
| faiss-cpu | 1.14+ | Vector similarity search |
| joblib | — | Model serialization |
| colorama | — | Colored terminal output |
| torch | 2.0+ | Backend for sentence-transformers |

### Performance

| Operation | Time | Notes |
|---|---|---|
| Model loading | ~2s | All 3 sklearn models |
| KB loading | ~3s | FAISS index + sentence-transformers model |
| Crop prediction | ~30ms | sklearn predict + rule engine + RAG search |
| Fertilizer advice | ~10ms | Rule engine + optional ML predict + RAG |
| RAG search | ~50ms | FAISS similarity search over 3,870 vectors |
| KB build | ~45s | One-time: encode 3,870 docs → 384-dim vectors |

### Limitations

1. **Not a generative LLM** — responses use templates, not free-form text generation
2. **English-focused** — entity extraction works best with English input
3. **Disease dataset limited** — only 2,000 of ~100K+ rows indexed for performance
4. **No image analysis** — cannot diagnose diseases from photos
5. **Offline only** — no API calls; all inference is local

### Future Enhancements

- Integrate a local LLM (Llama 3.1 / Mistral) for more natural response generation
- Add a Flask/FastAPI web interface
- Index the full 122MB disease dataset
- Add image-based disease diagnosis using CNN models
- Support Hindi/regional language input
- Add weather API integration for real-time forecasts
