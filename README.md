# Tracer AI Diagnostic Loop Tracking System

> **MedGemma Impact Challenge Submission**
> Preventing missed diagnoses in ambulatory care through fine-tuned MedGemma and a 4-agent quality pipeline.

**[Live Demo](https://tracer-health.vercel.app/)**

**[Fine-tuned Model (HuggingFace)](https://huggingface.co/adishofwhat/tracer-medgemma-v2)**

**[Fine-tuning Notebook (Kaggle)](https://www.kaggle.com/code/adishgolechha/medgemma-finetuning-final)**

---

## The Problem

A 58-year-old patient visits her primary care physician with three months of abdominal bloating and early satiety. Her doctor suspects ovarian pathology, orders a CA-125 and transvaginal ultrasound, and documents the clinical reasoning clearly. The CA-125 comes back at 685 U/mL — critically elevated, normal is under 35. The result lands in an inbox of 200 items. The ultrasound was never scheduled. Six weeks later, she presents with a bowel obstruction from advanced ovarian cancer.

The loop was never closed.

Diagnostic errors cause an estimated 795,000 deaths or permanent disabilities annually in the US (Newman-Toker et al., BMJ Quality & Safety, 2024). The majority are not failures of clinical knowledge — they are failures of workflow. A physician suspects the right diagnosis, orders the right test, and the result either never arrives or arrives without the context needed to act on it.

**Tracer solves the follow-up gap.** It reads the original clinical note at the time of ordering, extracts the diagnostic hypothesis using fine-tuned MedGemma, and monitors incoming results — alerting physicians when a critical finding arrives or when a pending test has gone too long without resolution.

---

## Solution Overview
```
Clinical Note → Fine-tuned MedGemma → Structured Hypothesis
                                              ↓
                                    4-Agent Quality Pipeline
                                              ↓
                              AI-Enriched Results Inbox + Loop Tracker
```

When a physician orders a test, Tracer:
1. Extracts the diagnostic hypothesis from the clinical note (primary hypothesis, differential, key symptoms, urgency, reasoning)
2. Runs the extraction through a 4-agent validation pipeline
3. Stores the structured hypothesis against the patient record
4. When results arrive, surfaces the original clinical context alongside the new finding
5. Flags open loops — tests ordered but never completed — for physician follow-up

---

## Technical Architecture

### Model
- **Base model:** MedGemma 1.5 (4B-it) — Google's medical-domain pretrained model
- **Fine-tuning:** LoRA adaptation on 421 high-quality clinical note → hypothesis extraction pairs
- **Training:** Kaggle GPU (T4), 1h 48m, prompt masking enabled
- **Result:** Validation loss 0.866 (19.4% improvement from baseline 1.075), 100% field completeness vs 75% for base MedGemma zero-shot

### 4-Agent Quality Pipeline
| Agent | Role | Model |
|---|---|---|
| Agent 1 | Rule-based pre-validator | Deterministic |
| Agent 2 | Hypothesis extractor | Fine-tuned MedGemma v2 |
| Agent 3 | Quality checker (urgency, plausibility) | Base MedGemma |
| Agent 4 | Confidence scorer + flag trigger | Base MedGemma |

Each patient record carries `agent_confidence` (1–10) and `agent_review_flag` (boolean) surfaced prominently in the physician UI. If the Confidence Scorer rates an extraction below 6/10, the output is flagged and the Quality Checker re-evaluates with an expanded validation prompt before the result reaches the physician — creating a self-correcting loop rather than a purely linear chain.

**Safety & Human-in-the-Loop:** Tracer is explicitly non-autonomous. It surfaces context — it does not make decisions. All open loops are surfaced to the physician regardless of confidence score; the model cannot suppress a result. Agent 4 flags low-confidence extractions (<6/10) for manual review rather than acting on them. The physician remains the decision-maker at every step. Audit trails of AI-generated context are stored alongside the result for accountability.

### Frontend
- **Framework:** Next.js 14, TypeScript, Tailwind CSS
- **Deployment:** Vercel
- **Design:** EHR-style interface — clinical, data-dense, zero decorative elements
- **Three views:** Results Inbox, Patient Detail, Loop Tracker

### Integration Path (Production)
The prototype uses pre-computed outputs. Production integration runs through standard FHIR R4 APIs that Epic and Cerner both expose — no custom EHR modification required.

---

## Demo

The live demo includes 20 realistic patient scenarios across multiple specialties — oncology, cardiology, neurology, emergency medicine — each representing a real failure mode where a diagnostic loop was not closed.

**Key cases to explore:**
- **P001** Ovarian cancer: CA-125 of 685, ultrasound never scheduled (18 days pending)
- **P011** Medulloblastoma in a 16-year-old: MRI delayed 10 days awaiting insurance authorization, Confidence 10/10
- **P010** DVT: D-dimer 2,850, ultrasound pending 5 days, PE risk
- **P020** Glioblastoma: patient delayed MRI for work project despite bilateral papilledema

---

## Repository Structure
```
tracer/
├── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Dataset analysis and validation
│   ├── 02_model_finetuning.ipynb       # MedGemma LoRA fine-tuning
│   ├── 03_batch_inference.ipynb        # Inference on patient scenarios
│   ├── 04_evaluation.ipynb             # Base vs fine-tuned comparison
│   └── 05_agentic_pipeline.ipynb       # 4-agent quality pipeline
├── src/
│   ├── ai/
│   │   ├── hypothesis_extractor.py     # Core extraction logic
│   │   ├── loop_detector.py            # Pending order detection
│   │   └── result_analyzer.py         # Result contextualization
│   └── utils/
│       ├── data_loader.py              # Data loading utilities
│       └── evaluator.py               # Model evaluation utilities
├── scripts/
│   ├── data_pipeline.py               # Data validation pipeline
│   ├── data_qa_pipeline.py            # QA checks on training data
│   ├── generate_patients.py           # Patient scenario generation
│   └── generate_training_data.py      # Training example generation
└── frontend/
    ├── data/
    │   └── patients_with_ai_final_enriched.json
    └── app/                            # Next.js application
```

---

## Setup

### Run the demo locally
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

### Run fine-tuning notebook

The fine-tuning notebook is designed for Kaggle (GPU T4). Open `notebooks/02_model_finetuning.ipynb` on Kaggle and attach the MedGemma model from the Kaggle model hub.

---

## Dataset

- **421 training examples** across 25 medical specialties
- **20 patient scenarios** for demo — diverse urgency levels, specialties, and failure modes
- All data is synthetic, generated with clinical accuracy validation
- Training data schema: `input` (clinical note) → `output` (6-field structured hypothesis)

---

## Model Performance

We evaluated Tracer v2 against 7 open-source models (≤7B parameters) on 10 held-out clinical notes:

| Model | Size | Completeness | Valid Structure | Urgency Accuracy |
|---|---|---|---|---|
| BioMistral 7B (medical, PubMed) | 7B | 17% | 0% | 0% |
| Meditron 7B (medical, guidelines) | 7B | 23% | 0% | 0% |
| Llama 3.2 3B (general) | 3B | 53% | 0% | 80% |
| Qwen2.5 7B (general) | 7B | 67% | 0% | 80% |
| Gemma 2 2B (general) | 2B | 92% | 100% | 60% |
| Base MedGemma 4B (zero-shot) | 4B | 75% | 50% | 90% |
| DeepSeek-R1 7B (reasoning) | 7B | 100% | 100% | 90% |
| **Tracer v2 (fine-tuned MedGemma)** | **4B** | **100%** | **100%** | **90%** |

Key finding: every dedicated medical LLM (BioMistral, Meditron) scores **0% valid structure** despite clinical pretraining — domain knowledge alone is insufficient. Fine-tuning over base MedGemma adds +25pp completeness and +50pp structural validity. Tracer v2 matches the strongest reasoning model in the comparison at half the parameter count.

---

## Impact

Structured follow-up interventions reduce missed care events by 20–30% in **[published studies](pmc.ncbi.nlm.nih.gov/articles/PMC12296817)**. Applying a conservative 15% improvement to diagnostic loop closure rates translates to ~119,000 serious harms prevented and ~$630M/year in avoided malpractice annually (Newman-Toker et al., BMJ Quality & Safety, 2024; Coverys, 2025).

The 23 open loops in the demo represent exactly the failure pattern that causes these outcomes — tests ordered, results arrived, follow-up never happened. Tracer closes the loop.

---

## Competition

Built for the **MedGemma Impact Challenge** (Google, February 2026).
Uses fine-tuned MedGemma 1.5 (4B-it) as the core extraction model.

## License
CC BY 4.0 — see [LICENSE](LICENSE) for details.
