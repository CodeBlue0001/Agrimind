"""
Knowledge Base Builder — RAG Pipeline for AgroCulture AI
=========================================================
Converts CSV datasets into a searchable vector knowledge base using
sentence-transformers for embeddings and FAISS for vector search.

Usage:
    python build_knowledge_base.py                  # Build from all datasets
    python build_knowledge_base.py --datasets fert  # Build only fertilizer KB
    python build_knowledge_base.py --test            # Build and run test queries

Output:
    model/llm/data/knowledge_chunks.json   — text chunks
    model/llm/embeddings/faiss_index/      — FAISS vector index
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Defer heavy imports to allow fast --help
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, "embeddings", "faiss_index")
DATASET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "dataset"))


# ---------------------------------------------------------------------------
# Document Generators — convert CSV rows to natural-language text chunks
# ---------------------------------------------------------------------------

def generate_crop_docs(csv_path: str) -> List[Dict]:
    """Generate knowledge documents from sensor_Crop_Dataset."""
    docs = []
    df = pd.read_csv(csv_path)
    print(f"   [crop] Loaded {len(df)} rows from {os.path.basename(csv_path)}")

    # Group by Crop + Soil_Type to create summary documents
    for (crop, soil), group in df.groupby(["Crop", "Soil_Type"]):
        n_mean = group["Nitrogen"].mean()
        p_mean = group["Phosphorus"].mean()
        k_mean = group["Potassium"].mean()
        temp_mean = group["Temperature"].mean()
        hum_mean = group["Humidity"].mean()
        ph_mean = group["pH_Value"].mean()
        rain_mean = group["Rainfall"].mean()

        varieties = group["Variety"].dropna().unique().tolist()
        variety_str = ", ".join([v for v in varieties if v][:5])

        text = (
            f"{crop} cultivation in {soil} soil: "
            f"Ideal conditions are Nitrogen={n_mean:.0f}, Phosphorus={p_mean:.0f}, "
            f"Potassium={k_mean:.0f}, Temperature={temp_mean:.1f}°C, "
            f"Humidity={hum_mean:.1f}%, pH={ph_mean:.2f}, Rainfall={rain_mean:.0f}mm. "
        )
        if variety_str:
            text += f"Common varieties: {variety_str}. "

        text += f"Based on {len(group)} data points."

        docs.append({
            "text": text,
            "source": "sensor_Crop_Dataset",
            "category": "crop_requirements",
            "crop": crop,
            "soil_type": soil,
        })

    print(f"   [crop] Generated {len(docs)} summary documents")
    return docs


def generate_fertilizer_docs(csv_path: str) -> List[Dict]:
    """Generate knowledge documents from Fertilizer Prediction dataset."""
    docs = []
    df = pd.read_csv(csv_path)
    print(f"   [fert] Loaded {len(df)} rows from {os.path.basename(csv_path)}")

    # Normalize column names
    col_map = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_")
        col_map[col] = clean
    df.rename(columns=col_map, inplace=True)

    # Determine fertilizer column name
    fert_col = "fertilizer" if "fertilizer" in df.columns else "fertilizer_name"

    # Group by Crop + Soil + Fertilizer
    group_cols = []
    if "crop_type" in df.columns:
        group_cols.append("crop_type")
    if "soil_type" in df.columns:
        group_cols.append("soil_type")
    if fert_col in df.columns:
        group_cols.append(fert_col)

    if len(group_cols) >= 2:
        for keys, group in df.groupby(group_cols):
            if len(group_cols) == 3:
                crop, soil, fert = keys
            else:
                crop, fert = keys[0], keys[-1]
                soil = "various"

            temp_mean = group.get("temparature", group.get("temperature", pd.Series([25]))).mean()
            hum_mean = group.get("humidity", pd.Series([60])).mean()

            n_mean = group.get("nitrogen", pd.Series([0])).mean()
            p_mean = group.get("phosphorous", group.get("phosphorus", pd.Series([0]))).mean()
            k_mean = group.get("potassium", pd.Series([0])).mean()

            text = (
                f"For {crop} grown in {soil} soil with Temperature={temp_mean:.0f}°C and "
                f"Humidity={hum_mean:.0f}%, when soil has Nitrogen={n_mean:.0f}, "
                f"Phosphorus={p_mean:.0f}, Potassium={k_mean:.0f}, "
                f"the recommended fertilizer is {fert}."
            )

            docs.append({
                "text": text,
                "source": os.path.basename(csv_path),
                "category": "fertilizer_recommendation",
                "crop": str(crop),
                "fertilizer": str(fert),
            })

    print(f"   [fert] Generated {len(docs)} documents")
    return docs


def generate_soil_fertility_docs(csv_path: str) -> List[Dict]:
    """Generate knowledge documents from soil_fertility_dataset."""
    docs = []
    df = pd.read_csv(csv_path)
    print(f"   [soil] Loaded {len(df)} rows from {os.path.basename(csv_path)}")

    # Classify fertility based on Output column (0=not fertile, 1=fertile)
    for _, row in df.iterrows():
        fertility = "fertile" if row.get("Output", 0) == 1 else "not fertile"
        text = (
            f"Soil sample with N={row.get('N', 0):.0f}, P={row.get('P', 0):.1f}, "
            f"K={row.get('K', 0):.0f}, pH={row.get('pH', 0):.2f}, "
            f"EC={row.get('EC', 0):.2f}, OC={row.get('OC', 0):.2f}, "
            f"S={row.get('S', 0):.1f}, Zn={row.get('Zn', 0):.2f}, "
            f"Fe={row.get('Fe', 0):.2f}, Cu={row.get('Cu', 0):.2f}, "
            f"Mn={row.get('Mn', 0):.2f}, B={row.get('B', 0):.2f} "
            f"is classified as {fertility}."
        )

        docs.append({
            "text": text,
            "source": "soil_fertility_dataset",
            "category": "soil_analysis",
            "fertility": fertility,
        })

    # Also generate summary statistics
    fertile = df[df["Output"] == 1]
    not_fertile = df[df["Output"] == 0]

    if len(fertile) > 0:
        summary_text = (
            f"Fertile soil characteristics (based on {len(fertile)} samples): "
            f"Average N={fertile['N'].mean():.0f}, P={fertile['P'].mean():.1f}, "
            f"K={fertile['K'].mean():.0f}, pH={fertile['pH'].mean():.2f}, "
            f"OC={fertile['OC'].mean():.2f}. "
            f"Fertile soils typically have higher organic carbon (>0.5%) and "
            f"balanced pH between 6.0-7.5."
        )
        docs.append({
            "text": summary_text,
            "source": "soil_fertility_dataset",
            "category": "soil_analysis_summary",
        })

    print(f"   [soil] Generated {len(docs)} documents")
    return docs


def generate_soil_climate_docs(csv_path: str) -> List[Dict]:
    """Generate knowledge documents from Soil-Climate-data."""
    docs = []
    df = pd.read_csv(csv_path)
    print(f"   [clim] Loaded {len(df)} rows from {os.path.basename(csv_path)}")

    # Group by Crop_Type + Soil_Type for summary
    for (crop, soil), group in df.groupby(["Crop_Type", "Soil_Type"]):
        compatible = group[group["Compatible"] == 1]
        incompatible = group[group["Compatible"] == 0]

        ph_mean = group["Soil_pH"].mean()
        n_mean = group["Soil_Nitrogen"].mean()
        om_mean = group["Soil_Organic_Matter"].mean()
        temp_mean = group["Temperature"].mean()
        rain_mean = group["Rainfall"].mean()
        hum_mean = group["Humidity"].mean()

        compat_rate = len(compatible) / len(group) * 100 if len(group) > 0 else 0

        text = (
            f"{crop} in {soil}: Compatibility rate {compat_rate:.0f}%. "
            f"Typical conditions: pH={ph_mean:.1f}, Nitrogen={n_mean:.0f}, "
            f"Organic Matter={om_mean:.1f}%, Temperature={temp_mean:.1f}°C, "
            f"Rainfall={rain_mean:.0f}mm, Humidity={hum_mean:.0f}%. "
            f"Based on {len(group)} field observations."
        )

        docs.append({
            "text": text,
            "source": "Soil-Climate-data",
            "category": "crop_soil_compatibility",
            "crop": str(crop),
            "soil_type": str(soil),
        })

    print(f"   [clim] Generated {len(docs)} documents")
    return docs


def generate_rainfall_docs(csv_path: str) -> List[Dict]:
    """Generate knowledge documents from merged_rainfall_dataset."""
    docs = []
    df = pd.read_csv(csv_path)
    print(f"   [rain] Loaded {len(df)} rows from {os.path.basename(csv_path)}")

    for _, row in df.iterrows():
        state = row.get("State", "")
        district = row.get("District", "")
        avg_rain = row.get("Avg_rainfall", 0)

        if avg_rain > 0:
            text = (
                f"Average annual rainfall in {district}, {state} is {avg_rain:.1f} mm. "
            )
            # Classify rainfall
            if avg_rain < 500:
                text += "This is a low rainfall region, suitable for drought-resistant crops like bajra, jowar, and pulses."
            elif avg_rain < 1000:
                text += "This is a moderate rainfall region, suitable for wheat, mustard, and oil seeds."
            elif avg_rain < 2000:
                text += "This is a high rainfall region, suitable for rice, jute, and sugarcane."
            else:
                text += "This is a very high rainfall region, suitable for tea, rubber, and tropical fruits."

            docs.append({
                "text": text,
                "source": "merged_rainfall_dataset",
                "category": "rainfall_info",
                "state": str(state),
                "district": str(district),
            })

    print(f"   [rain] Generated {len(docs)} documents")
    return docs


def generate_disease_docs(csv_path: str, max_rows: int = 2000) -> List[Dict]:
    """
    Generate knowledge documents from crop_disease_remediation.
    Limited to max_rows to keep the index manageable.
    """
    docs = []

    try:
        df = pd.read_csv(csv_path, nrows=max_rows, encoding='utf-8', on_bad_lines='skip')
    except Exception:
        try:
            df = pd.read_csv(csv_path, nrows=max_rows, encoding='latin-1', on_bad_lines='skip')
        except Exception as e:
            print(f"   [disease] Warning: Could not read {csv_path}: {e}")
            return docs

    print(f"   [disease] Loaded {len(df)} rows from {os.path.basename(csv_path)} (max {max_rows})")

    # Use English columns if available
    q_col = "question_en" if "question_en" in df.columns else df.columns[0]
    r_col = "remedy_en" if "remedy_en" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)

    for col in ["crop_name_en", "disease_name_en", "disease_type", "season", "severity_level"]:
        if col not in df.columns:
            df[col] = ""

    for _, row in df.iterrows():
        question = str(row.get(q_col, "")).strip()
        remedy = str(row.get(r_col, "")).strip() if r_col else ""
        crop = str(row.get("crop_name_en", "")).strip()
        disease = str(row.get("disease_name_en", "")).strip()
        disease_type = str(row.get("disease_type", "")).strip()
        season = str(row.get("season", "")).strip()
        severity = str(row.get("severity_level", "")).strip()

        # Skip rows with non-English or empty content
        if not question or len(question) < 10:
            continue

        # Try to use enhanced_prompt/completion if available
        enhanced_q = str(row.get("enhanced_prompt", "")).strip()
        enhanced_r = str(row.get("enhanced_completion", "")).strip()

        if enhanced_q and len(enhanced_q) > 20:
            text = enhanced_q
            if enhanced_r and len(enhanced_r) > 20:
                # Truncate very long completions
                text += " " + enhanced_r[:500]
        else:
            text = f"Q: {question}"
            if remedy:
                text += f" A: {remedy[:500]}"

        # Add metadata context
        meta_parts = []
        if crop:
            meta_parts.append(f"Crop: {crop}")
        if disease:
            meta_parts.append(f"Disease: {disease}")
        if disease_type:
            meta_parts.append(f"Type: {disease_type}")
        if season:
            meta_parts.append(f"Season: {season}")
        if severity:
            meta_parts.append(f"Severity: {severity}")

        if meta_parts:
            text = " | ".join(meta_parts) + " | " + text

        docs.append({
            "text": text[:1000],  # Cap at 1000 chars per doc
            "source": "crop_disease_remediation",
            "category": "disease_info",
            "crop": crop,
            "disease": disease,
        })

    print(f"   [disease] Generated {len(docs)} documents")
    return docs


# ---------------------------------------------------------------------------
# Embedding & Index Building
# ---------------------------------------------------------------------------

def build_index(documents: List[Dict], batch_size: int = 64) -> None:
    """
    Build FAISS index from documents using sentence-transformers.

    Args:
        documents: List of dicts with at least a 'text' key
        batch_size: Batch size for encoding
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"\n[INDEX] Building vector index for {len(documents)} documents...")

    # Save documents
    os.makedirs(DATA_DIR, exist_ok=True)
    chunks_path = os.path.join(DATA_DIR, "knowledge_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=1)
    print(f"   [SAVE] Chunks saved to: {chunks_path}")

    # Load sentence transformer model
    print("   [MODEL] Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode all texts
    texts = [doc["text"] for doc in documents]
    print(f"   [ENCODE] Encoding {len(texts)} documents (batch_size={batch_size}) ...")

    start = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    elapsed = time.time() - start
    print(f"   [ENCODE] Done in {elapsed:.1f}s — embedding shape: {embeddings.shape}")

    # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save index
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    index_path = os.path.join(EMBEDDINGS_DIR, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"   [SAVE] FAISS index saved to: {index_path}")
    print(f"   [SAVE] Index contains {index.ntotal} vectors of dimension {dim}")


# ---------------------------------------------------------------------------
# Search Interface
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Search interface for the built knowledge base.
    """

    def __init__(self):
        self.documents = None
        self.index = None
        self.model = None
        self._loaded = False

    def load(self) -> bool:
        """Load the knowledge base from disk."""
        import faiss

        chunks_path = os.path.join(DATA_DIR, "knowledge_chunks.json")
        index_path = os.path.join(EMBEDDINGS_DIR, "index.faiss")

        if not os.path.exists(chunks_path) or not os.path.exists(index_path):
            print("[WARN] Knowledge base not found. Run build_knowledge_base.py first.")
            return False

        print("[KB] Loading knowledge base...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.index = faiss.read_index(index_path)

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self._loaded = True
        print(f"   [OK] {len(self.documents)} documents, {self.index.ntotal} vectors")
        return True

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict]:
        """
        Search the knowledge base with a natural language query.

        Args:
            query: Search query
            top_k: Number of results to return
            category: Optional category filter

        Returns:
            List of matching documents with scores
        """
        if not self._loaded:
            if not self.load():
                return []

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx].copy()
            doc["score"] = float(score)

            # Apply category filter
            if category and doc.get("category") != category:
                continue

            results.append(doc)

            if len(results) >= top_k:
                break

        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def stats(self) -> Dict:
        if not self._loaded:
            return {"loaded": False}

        categories = {}
        for doc in self.documents:
            cat = doc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "loaded": True,
            "total_documents": len(self.documents),
            "total_vectors": self.index.ntotal,
            "categories": categories,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build AgroCulture AI Knowledge Base")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Which datasets to include: crop fert soil climate rain disease (default: all)")
    parser.add_argument("--disease-rows", type=int, default=2000,
                        help="Max rows from disease dataset (default: 2000)")
    parser.add_argument("--test", action="store_true",
                        help="Run test queries after building")
    args = parser.parse_args()

    print("=" * 70)
    print("  AgroCulture AI — Knowledge Base Builder")
    print("=" * 70)
    print(f"  Dataset dir: {DATASET_DIR}")
    print(f"  Output dir:  {DATA_DIR}")
    print()

    include_all = args.datasets is None
    datasets = set(args.datasets) if args.datasets else set()

    all_docs = []

    # 1. Crop dataset
    crop_path = os.path.join(DATASET_DIR, "sensor_Crop_Dataset (1).csv")
    if os.path.exists(crop_path) and (include_all or "crop" in datasets):
        all_docs.extend(generate_crop_docs(crop_path))

    # 2. Fertilizer datasets
    for fname in ["Fertilizer Prediction.csv", "f2.csv"]:
        fert_path = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fert_path) and (include_all or "fert" in datasets):
            all_docs.extend(generate_fertilizer_docs(fert_path))

    # 3. Soil fertility dataset
    soil_path = os.path.join(DATASET_DIR, "soil_fertility_dataset.csv")
    if os.path.exists(soil_path) and (include_all or "soil" in datasets):
        all_docs.extend(generate_soil_fertility_docs(soil_path))

    # 4. Soil-climate dataset
    climate_path = os.path.join(DATASET_DIR, "Soil-Climate-data.csv")
    if os.path.exists(climate_path) and (include_all or "climate" in datasets):
        all_docs.extend(generate_soil_climate_docs(climate_path))

    # 5. Rainfall dataset
    rain_path = os.path.join(DATASET_DIR, "merged_rainfall_dataset.csv")
    if os.path.exists(rain_path) and (include_all or "rain" in datasets):
        all_docs.extend(generate_rainfall_docs(rain_path))

    # 6. Disease dataset
    disease_path = os.path.join(DATASET_DIR, "crop_disease_remediation.csv")
    if os.path.exists(disease_path) and (include_all or "disease" in datasets):
        all_docs.extend(generate_disease_docs(disease_path, max_rows=args.disease_rows))

    if not all_docs:
        print("\n[ERROR] No documents generated. Check dataset paths.")
        sys.exit(1)

    print(f"\n[TOTAL] {len(all_docs)} documents generated")

    # Build index
    build_index(all_docs)

    print("\n" + "=" * 70)
    print("  Knowledge Base Built Successfully!")
    print("=" * 70)

    # Test search
    if args.test:
        print("\n[TEST] Running test queries...\n")
        kb = KnowledgeBase()
        kb.load()

        test_queries = [
            "What fertilizer for rice in loamy soil?",
            "Soil with pH 7.4 and low nitrogen",
            "White rot disease treatment",
            "Best conditions for sugarcane cultivation",
            "Rainfall in Punjab",
        ]

        for q in test_queries:
            print(f"\nQuery: {q}")
            results = kb.search(q, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['score']:.3f}] {r['text'][:120]}...")
            print("-" * 60)


if __name__ == "__main__":
    main()
