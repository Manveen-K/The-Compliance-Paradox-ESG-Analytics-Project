# 🌍 The-Compliance-Paradox
A Forensic NLP Analysis of ESG Narrative Divergence in European Energy Reporting (2019–2025)
### ESG Analytics · Group 7

A reproducible **three-layer NLP pipeline** that forensically distinguishes genuine ESG performance from narrative inflation in corporate sustainability reports - applied to **Shell plc**, **BP plc**, and **Ørsted A/S** across 21 company-year observations (2019–2025).

---

## 📌 Overview

Corporate sustainability reporting has proliferated alongside rising accusations of greenwashing. Existing ESG rating methodologies treat disclosure *volume* as a proxy for disclosure *quality*, rewarding narrative ambition without anchoring it to operational performance. This project addresses that gap by combining transformer-based sentiment analysis, ensemble materiality classification, and keyword density drift — all grounded in verified physical emissions data.

**Central question:** Can a reproducible NLP pipeline forensically distinguish genuine ESG performance from narrative inflation within a cohort of ostensibly comparable companies?

---

## 🏗️ Framework Architecture

The pipeline consists of three independent analytical layers:

### Layer 1 — Narrative Divergence Index (NDI)
Plots narrative tone directly against physical emissions performance.

```
NDI = (ARG + EPD) / 2  ∈ (-1, +1)

ARG = (NOS_Aspirational − NOS_Operational) / 2   [rhetoric gap]
EPD = −tanh(ΔEmissions_YoY / 5)                  [reality anchor]
```

A **negative NDI** is the target: physical reductions outpace narrative. A **positive NDI** signals rhetoric exceeds delivery.

### Layer 2 — Double Materiality Balance (DM_Balance)
A three-classifier majority-vote ensemble (spaCy SVO + FinBERT-ESG-9 + vocabulary matcher) classifies each sentence as **Impact Materiality** (inside-out) or **Financial Materiality** (outside-in).

```
DM_Balance = |Impact% − Financial%|

CSRD-adjacent bands: GREEN ≤25pp | YELLOW 26–35pp | ORANGE 36–50pp | RED >50pp
```

### Layer 3 — Keyword Density Drift (GR + NVI)
Tracks the evolution of aspirational vs. operational vocabulary per 1,000 words using KeyBERT (MiniLM-L6-v2).

```
Greenwashing Ratio (GR)  = Commitment_density / Operational_density        [unbounded]
NVI                       = (Commit − Ops) / (Commit + Ops + ε)  ∈ (-1,+1) [scale-invariant]
```

---

## 📊 Key Findings

| Company | Greenwashing Archetype | Mean NDI | Emissions Change |
|---|---|---|---|
| **Shell plc** | Narrative Collapse | Volatile (−0.31 to +0.28) | −9.0% (78→71 gCO₂e/MJ) |
| **BP plc** | Vocabulary Suppression | +0.09 (positive 5/7 years) | −0.9% (79.7→79.0 gCO₂e/MJ) |
| **Ørsted A/S** | Genuine Performance ✅ | **−0.257** (negative 6/7 years) | **−93.8%** (65→4 gCO₂e/kWh) |

**CSRD Compliance Paradox:** All three companies achieved GREEN-band Double Materiality Balance in every single reporting year - despite exhibiting radically divergent disclosure behaviours. Balance-based regulatory metrics are structurally blind to the narrative mechanisms this framework detects.

---

## 📦 Dependencies

Installed automatically in Cell 1 of each notebook:

```
pdfplumber · transformers>=4.38 · torch · sentencepiece · spacy (en_core_web_lg)
scipy · matplotlib · seaborn · nltk · tqdm · accelerate · pandas · numpy
pyarrow · keybert · sentence-transformers
```

Models used: **ClimateBERT** (narrative scoring) · **FinBERT-ESG-9** (materiality classification) · **MiniLM-L6-v2** via KeyBERT (keyword extraction)

---

## 📄 Data Sources

- Shell, BP, and Ørsted annual **Sustainability Reports** (2019–2025), sourced from official corporate repositories.
- Carbon intensity data transcribed from ESG data books and cross-verified against investor presentations.
- 20,905 ESG sentences total: Shell=9,046, BP=6,164, Ørsted=5,695.


