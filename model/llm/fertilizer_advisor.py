"""
Fertilizer Advisor — Rule-Based Dose Calculator
=================================================
Given current soil NPK data and a target crop, calculates nutrient
deficits and recommends specific commercial fertilizers with dosages.

Uses ideal NPK ranges derived from agronomic standards and the
sensor_Crop_Dataset statistics.
"""

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Ideal NPK Ranges by Crop (kg/ha or ppm equivalent)
# ---------------------------------------------------------------------------
# These are typical recommended ranges for Indian agricultural conditions.
# Sources: ICAR recommendations, sensor_Crop_Dataset statistics, and
# standard agronomic guidelines.

CROP_NPK_IDEALS = {
    "Rice":       {"N": (80, 120),  "P": (40, 60),  "K": (40, 80)},
    "Wheat":      {"N": (80, 120),  "P": (40, 60),  "K": (30, 50)},
    "Maize":      {"N": (100, 150), "P": (50, 75),  "K": (40, 60)},
    "Sugarcane":  {"N": (120, 180), "P": (50, 80),  "K": (60, 100)},
    "Cotton":     {"N": (60, 100),  "P": (30, 50),  "K": (30, 50)},
    "Tea":        {"N": (100, 150), "P": (30, 50),  "K": (80, 120)},
    "Coffee":     {"N": (80, 120),  "P": (60, 80),  "K": (80, 120)},
    "Potato":     {"N": (100, 150), "P": (60, 80),  "K": (80, 120)},
    "Tomato":     {"N": (100, 140), "P": (60, 80),  "K": (60, 100)},
    "Onion":      {"N": (80, 120),  "P": (40, 60),  "K": (60, 80)},
    "Groundnut":  {"N": (20, 40),   "P": (40, 60),  "K": (40, 60)},
    "Soybean":    {"N": (20, 40),   "P": (60, 80),  "K": (30, 50)},
    "Mustard":    {"N": (60, 80),   "P": (30, 40),  "K": (20, 30)},
    "Sunflower":  {"N": (60, 90),   "P": (40, 60),  "K": (30, 50)},
    "Jute":       {"N": (40, 60),   "P": (20, 30),  "K": (30, 40)},
    "Tobacco":    {"N": (60, 100),  "P": (30, 50),  "K": (80, 120)},
    "Coconut":    {"N": (50, 80),   "P": (30, 50),  "K": (80, 120)},
    "Banana":     {"N": (100, 150), "P": (30, 50),  "K": (150, 200)},
    "Mango":      {"N": (80, 120),  "P": (40, 60),  "K": (80, 120)},
    "Grape":      {"N": (80, 120),  "P": (60, 80),  "K": (80, 120)},
    "Chilli":     {"N": (80, 120),  "P": (40, 60),  "K": (40, 60)},
    "Turmeric":   {"N": (60, 100),  "P": (30, 50),  "K": (80, 120)},
    "Ginger":     {"N": (75, 100),  "P": (50, 75),  "K": (50, 75)},
    "Barley":     {"N": (60, 80),   "P": (30, 40),  "K": (20, 30)},
    "Peas":       {"N": (20, 30),   "P": (40, 60),  "K": (30, 50)},
    "Lentil":     {"N": (20, 30),   "P": (40, 60),  "K": (20, 30)},
    "Chickpea":   {"N": (20, 30),   "P": (40, 60),  "K": (20, 30)},
    "Gram":       {"N": (20, 30),   "P": (40, 60),  "K": (20, 30)},
    "Bajra":      {"N": (60, 80),   "P": (30, 40),  "K": (20, 30)},
    "Jowar":      {"N": (60, 80),   "P": (30, 40),  "K": (20, 30)},
    "Ragi":       {"N": (50, 75),   "P": (30, 40),  "K": (25, 40)},
    "Arhar":      {"N": (20, 30),   "P": (40, 60),  "K": (20, 30)},
    "Rubber":     {"N": (30, 50),   "P": (30, 50),  "K": (30, 50)},
    "Pepper":     {"N": (50, 100),  "P": (30, 50),  "K": (100, 150)},
    "Cardamom":   {"N": (75, 100),  "P": (75, 100), "K": (75, 100)},
    "Cabbage":    {"N": (100, 150), "P": (50, 80),  "K": (60, 100)},
    "Cauliflower":{"N": (100, 150), "P": (50, 80),  "K": (60, 100)},
    "Brinjal":    {"N": (80, 120),  "P": (50, 70),  "K": (50, 80)},
    "Watermelon": {"N": (80, 120),  "P": (40, 60),  "K": (60, 100)},
    "Papaya":     {"N": (100, 150), "P": (100, 150),"K": (100, 200)},
}

# Default for unknown crops
DEFAULT_NPK_IDEAL = {"N": (80, 120), "P": (40, 60), "K": (40, 60)}


# ---------------------------------------------------------------------------
# Commercial Fertilizers
# ---------------------------------------------------------------------------

FERTILIZER_DB = {
    "Urea": {
        "composition": {"N": 46.0, "P": 0, "K": 0},
        "description": "Most common nitrogen fertilizer (46% N)",
        "application": "Apply in 2-3 split doses during crop growth stages",
        "cost_inr_per_kg": 6.0,
    },
    "DAP (Di-Ammonium Phosphate)": {
        "composition": {"N": 18.0, "P": 46.0, "K": 0},
        "description": "High phosphorus fertilizer with some nitrogen (18-46-0)",
        "application": "Apply as basal dose at sowing/planting",
        "cost_inr_per_kg": 27.0,
    },
    "SSP (Single Super Phosphate)": {
        "composition": {"N": 0, "P": 16.0, "K": 0},
        "description": "Phosphorus fertilizer with calcium and sulphur (0-16-0)",
        "application": "Apply as basal dose, also provides sulphur",
        "cost_inr_per_kg": 9.0,
    },
    "MOP (Muriate of Potash)": {
        "composition": {"N": 0, "P": 0, "K": 60.0},
        "description": "Primary potassium fertilizer (0-0-60)",
        "application": "Apply as basal dose or in 2 splits",
        "cost_inr_per_kg": 17.0,
    },
    "NPK 10-26-26": {
        "composition": {"N": 10.0, "P": 26.0, "K": 26.0},
        "description": "Complex fertilizer for balanced P and K with some N",
        "application": "Apply as basal dose",
        "cost_inr_per_kg": 25.0,
    },
    "NPK 14-35-14": {
        "composition": {"N": 14.0, "P": 35.0, "K": 14.0},
        "description": "High-phosphorus complex fertilizer",
        "application": "Apply as basal dose, good for flowering crops",
        "cost_inr_per_kg": 26.0,
    },
    "NPK 20-20-0": {
        "composition": {"N": 20.0, "P": 20.0, "K": 0},
        "description": "Balanced N-P fertilizer",
        "application": "Apply as basal dose",
        "cost_inr_per_kg": 22.0,
    },
    "NPK 28-28-0": {
        "composition": {"N": 28.0, "P": 28.0, "K": 0},
        "description": "High N-P complex for N and P deficient soils",
        "application": "Apply as basal dose at sowing",
        "cost_inr_per_kg": 24.0,
    },
    "Ammonium Sulphate": {
        "composition": {"N": 21.0, "P": 0, "K": 0},
        "description": "Nitrogen fertilizer with 24% sulphur (21-0-0 + 24S)",
        "application": "Preferred for sulphur-deficient soils and tea gardens",
        "cost_inr_per_kg": 10.0,
    },
    "Potassium Sulphate (SOP)": {
        "composition": {"N": 0, "P": 0, "K": 50.0},
        "description": "Potassium fertilizer without chlorine (0-0-50 + 18S)",
        "application": "For chloride-sensitive crops like potato, tobacco, fruits",
        "cost_inr_per_kg": 40.0,
    },
}


# ---------------------------------------------------------------------------
# Soil Fertility Classification
# ---------------------------------------------------------------------------

SOIL_FERTILITY_RANGES = {
    "N": {"low": (0, 50), "medium": (50, 100), "high": (100, 999)},
    "P": {"low": (0, 25), "medium": (25, 55), "high": (55, 999)},
    "K": {"low": (0, 30), "medium": (30, 60), "high": (60, 999)},
    "pH": {"acidic": (0, 5.5), "slightly_acidic": (5.5, 6.5), "neutral": (6.5, 7.5),
           "slightly_alkaline": (7.5, 8.5), "alkaline": (8.5, 14)},
    "OC": {"low": (0, 0.5), "medium": (0.5, 0.75), "high": (0.75, 99)},
}


# ---------------------------------------------------------------------------
# Fertilizer Advisor
# ---------------------------------------------------------------------------

class FertilizerAdvisor:
    """
    Calculates NPK deficits and recommends commercial fertilizers
    with dosages based on current soil data and target crop.
    """

    def analyze_soil(self, soil_data: Dict) -> Dict:
        """
        Analyze soil health based on NPK, pH, and organic carbon values.

        Args:
            soil_data: Dict with keys like N, P, K, pH, OC (organic carbon)

        Returns:
            Dict with fertility classification for each parameter
        """
        analysis = {}

        for nutrient in ["N", "P", "K"]:
            value = soil_data.get(nutrient) or soil_data.get(nutrient.lower())
            if value is not None:
                value = float(value)
                ranges = SOIL_FERTILITY_RANGES[nutrient]
                status = "unknown"
                for level, (low, high) in ranges.items():
                    if low <= value < high:
                        status = level
                        break
                analysis[nutrient] = {"value": value, "status": status}

        # pH analysis
        ph = soil_data.get("pH") or soil_data.get("ph")
        if ph is not None:
            ph = float(ph)
            ph_ranges = SOIL_FERTILITY_RANGES["pH"]
            ph_status = "unknown"
            for level, (low, high) in ph_ranges.items():
                if low <= ph < high:
                    ph_status = level
                    break
            analysis["pH"] = {"value": ph, "status": ph_status}

        # Organic carbon
        oc = soil_data.get("OC") or soil_data.get("oc") or soil_data.get("organic_carbon")
        if oc is not None:
            oc = float(oc)
            oc_ranges = SOIL_FERTILITY_RANGES["OC"]
            oc_status = "unknown"
            for level, (low, high) in oc_ranges.items():
                if low <= oc < high:
                    oc_status = level
                    break
            analysis["OC"] = {"value": oc, "status": oc_status}

        return analysis

    def calculate_deficit(self, soil_data: Dict, crop: str) -> Dict:
        """
        Calculate NPK deficit for a target crop.

        Args:
            soil_data: Dict with N, P, K values (current soil levels)
            crop: Target crop name

        Returns:
            Dict with deficit amounts for N, P, K
        """
        # Normalize crop name for lookup
        crop_title = crop.strip().title()
        ideals = CROP_NPK_IDEALS.get(crop_title, DEFAULT_NPK_IDEAL)

        deficits = {}
        for nutrient in ["N", "P", "K"]:
            current = soil_data.get(nutrient) or soil_data.get(nutrient.lower(), 0)
            current = float(current)
            ideal_low, ideal_high = ideals[nutrient]
            ideal_mid = (ideal_low + ideal_high) / 2

            deficit = max(0, ideal_mid - current)
            status = "sufficient" if current >= ideal_low else "deficient"
            if current > ideal_high:
                status = "excess"

            deficits[nutrient] = {
                "current": current,
                "ideal_range": (ideal_low, ideal_high),
                "ideal_mid": ideal_mid,
                "deficit": deficit,
                "status": status,
            }

        return deficits

    def recommend_fertilizers(self, deficits: Dict, area_hectares: float = 1.0) -> List[Dict]:
        """
        Recommend commercial fertilizers based on NPK deficits.

        Args:
            deficits: Output from calculate_deficit()
            area_hectares: Farm area in hectares (default 1 ha)

        Returns:
            List of fertilizer recommendations with quantities
        """
        n_deficit = deficits.get("N", {}).get("deficit", 0)
        p_deficit = deficits.get("P", {}).get("deficit", 0)
        k_deficit = deficits.get("K", {}).get("deficit", 0)

        recommendations = []

        # If all deficits are small, no fertilizer needed
        if n_deficit < 5 and p_deficit < 5 and k_deficit < 5:
            return [{
                "fertilizer": "None required",
                "reason": "Your soil nutrient levels are within the ideal range for this crop.",
                "quantity_kg_per_ha": 0,
                "total_quantity_kg": 0,
            }]

        # Strategy: Use DAP for P (it also provides some N), then Urea for remaining N, MOP for K

        # Step 1: Address P deficit with DAP
        dap_needed = 0
        n_from_dap = 0
        if p_deficit > 5:
            dap_composition = FERTILIZER_DB["DAP (Di-Ammonium Phosphate)"]["composition"]
            dap_needed = p_deficit / (dap_composition["P"] / 100)  # kg per ha
            n_from_dap = dap_needed * (dap_composition["N"] / 100)

            recommendations.append({
                "fertilizer": "DAP (Di-Ammonium Phosphate)",
                "reason": f"Phosphorus deficit of {p_deficit:.1f} kg/ha",
                "quantity_kg_per_ha": round(dap_needed, 1),
                "total_quantity_kg": round(dap_needed * area_hectares, 1),
                "provides": f"P: {p_deficit:.1f} kg + N: {n_from_dap:.1f} kg per hectare",
                "application": FERTILIZER_DB["DAP (Di-Ammonium Phosphate)"]["application"],
                "estimated_cost_per_ha": round(dap_needed * FERTILIZER_DB["DAP (Di-Ammonium Phosphate)"]["cost_inr_per_kg"], 0),
            })

        # Step 2: Address remaining N deficit with Urea
        remaining_n = max(0, n_deficit - n_from_dap)
        if remaining_n > 5:
            urea_composition = FERTILIZER_DB["Urea"]["composition"]
            urea_needed = remaining_n / (urea_composition["N"] / 100)

            recommendations.append({
                "fertilizer": "Urea",
                "reason": f"Remaining nitrogen deficit of {remaining_n:.1f} kg/ha",
                "quantity_kg_per_ha": round(urea_needed, 1),
                "total_quantity_kg": round(urea_needed * area_hectares, 1),
                "provides": f"N: {remaining_n:.1f} kg per hectare",
                "application": FERTILIZER_DB["Urea"]["application"],
                "estimated_cost_per_ha": round(urea_needed * FERTILIZER_DB["Urea"]["cost_inr_per_kg"], 0),
            })

        # Step 3: Address K deficit with MOP
        if k_deficit > 5:
            mop_composition = FERTILIZER_DB["MOP (Muriate of Potash)"]["composition"]
            mop_needed = k_deficit / (mop_composition["K"] / 100)

            recommendations.append({
                "fertilizer": "MOP (Muriate of Potash)",
                "reason": f"Potassium deficit of {k_deficit:.1f} kg/ha",
                "quantity_kg_per_ha": round(mop_needed, 1),
                "total_quantity_kg": round(mop_needed * area_hectares, 1),
                "provides": f"K: {k_deficit:.1f} kg per hectare",
                "application": FERTILIZER_DB["MOP (Muriate of Potash)"]["application"],
                "estimated_cost_per_ha": round(mop_needed * FERTILIZER_DB["MOP (Muriate of Potash)"]["cost_inr_per_kg"], 0),
            })

        return recommendations

    def get_full_recommendation(self, soil_data: Dict, crop: str,
                                 area_hectares: float = 1.0) -> Dict:
        """
        Complete fertilizer recommendation pipeline.

        Args:
            soil_data: Dict with N, P, K (and optionally pH, OC)
            crop: Target crop name
            area_hectares: Farm area

        Returns:
            Complete recommendation with soil analysis, deficits, and fertilizers
        """
        soil_analysis = self.analyze_soil(soil_data)
        deficits = self.calculate_deficit(soil_data, crop)
        fertilizer_recs = self.recommend_fertilizers(deficits, area_hectares)

        total_cost = sum(r.get("estimated_cost_per_ha", 0) for r in fertilizer_recs)

        return {
            "crop": crop,
            "area_hectares": area_hectares,
            "soil_analysis": soil_analysis,
            "nutrient_deficits": deficits,
            "fertilizer_recommendations": fertilizer_recs,
            "total_estimated_cost_per_ha": round(total_cost, 0),
            "total_estimated_cost": round(total_cost * area_hectares, 0),
        }

    def format_recommendation(self, result: Dict) -> str:
        """
        Format the recommendation as a readable string.
        """
        lines = []
        crop = result["crop"]
        area = result["area_hectares"]

        lines.append(f"🌾 Fertilizer Recommendation for {crop}")
        lines.append(f"   Farm Area: {area} hectare(s)")
        lines.append("")

        # Soil Analysis
        lines.append("📊 Soil Analysis:")
        for nutrient, info in result["soil_analysis"].items():
            status_emoji = {"low": "🔴", "medium": "🟡", "high": "🟢",
                           "acidic": "🔴", "slightly_acidic": "🟡", "neutral": "🟢",
                           "slightly_alkaline": "🟡", "alkaline": "🔴",
                           "sufficient": "🟢", "deficient": "🔴", "excess": "🟡"
                           }.get(info["status"], "⚪")
            lines.append(f"   {status_emoji} {nutrient}: {info['value']} ({info['status']})")

        lines.append("")

        # Nutrient Deficits
        lines.append(f"📉 Nutrient Requirements for {crop}:")
        for nutrient, info in result["nutrient_deficits"].items():
            status_emoji = {"sufficient": "✅", "deficient": "⚠️", "excess": "📈"}.get(info["status"], "")
            ideal_range = f"{info['ideal_range'][0]}-{info['ideal_range'][1]}"
            lines.append(f"   {status_emoji} {nutrient}: Current={info['current']:.1f}, "
                        f"Ideal={ideal_range}, Deficit={info['deficit']:.1f}")

        lines.append("")

        # Fertilizer Recommendations
        recs = result["fertilizer_recommendations"]
        if recs and recs[0]["fertilizer"] != "None required":
            lines.append("💊 Recommended Fertilizers:")
            for i, rec in enumerate(recs, 1):
                lines.append(f"\n   {i}. {rec['fertilizer']}")
                lines.append(f"      Reason: {rec['reason']}")
                lines.append(f"      Dose: {rec['quantity_kg_per_ha']} kg/hectare")
                if area > 1:
                    lines.append(f"      Total for {area} ha: {rec['total_quantity_kg']} kg")
                lines.append(f"      Provides: {rec['provides']}")
                lines.append(f"      When: {rec['application']}")
                if rec.get('estimated_cost_per_ha'):
                    lines.append(f"      Est. Cost: ₹{rec['estimated_cost_per_ha']}/hectare")
        else:
            lines.append("✅ Your soil nutrient levels are already ideal for this crop!")
            lines.append("   No additional fertilizer is needed.")

        if result.get("total_estimated_cost_per_ha", 0) > 0:
            lines.append(f"\n   💰 Total Estimated Cost: ₹{result['total_estimated_cost_per_ha']}/hectare")
            if area > 1:
                lines.append(f"      Total for {area} ha: ₹{result['total_estimated_cost']}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_advisor = FertilizerAdvisor()

def get_recommendation(soil_data: Dict, crop: str, area: float = 1.0) -> str:
    """Get formatted fertilizer recommendation."""
    result = _advisor.get_full_recommendation(soil_data, crop, area)
    return _advisor.format_recommendation(result)

def analyze_soil(soil_data: Dict) -> Dict:
    """Analyze soil health."""
    return _advisor.analyze_soil(soil_data)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Fertilizer Advisor Test")
    print("=" * 70)

    # Test 1: Rice with low N
    soil1 = {"N": 40, "P": 10, "K": 80, "pH": 6.5}
    print("\n" + get_recommendation(soil1, "Sugarcane", 2.0))

    print("\n" + "=" * 70)

    # Test 2: Wheat with balanced soil
    soil2 = {"N": 100, "P": 50, "K": 45, "pH": 7.0}
    print("\n" + get_recommendation(soil2, "Wheat"))

    print("\n" + "=" * 70)

    # Test 3: Soil analysis only
    soil3 = {"N": 138, "P": 8.6, "K": 560, "pH": 7.46, "OC": 0.7}
    analysis = analyze_soil(soil3)
    print("\nSoil Analysis:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
