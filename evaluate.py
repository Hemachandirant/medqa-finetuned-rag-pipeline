"""
RAGAs Evaluation Suite — Medical RAG Pipeline
==============================================
Metrics selected:
  1. Faithfulness          — Hallucination guard (critical for clinical safety)
  2. Answer Relevancy      — Measures if answer actually addresses the question
  3. Context Precision     — Are retrieved chunks genuinely useful? (cost of noise)
  4. Context Recall        — Did we retrieve all necessary medical evidence?
  5. Answer Correctness    — End-to-end factual accuracy vs. ground truth

See README.md for full rationale.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.metrics.critique import harmfulness   # Safety metric — important for medical domain
import pandas as pd


# ─────────────────────────────────────────────
# Medical Ground Truth Test Cases
# ─────────────────────────────────────────────

MEDICAL_TEST_CASES = [
    # ── Factual ──────────────────────────────────────────────────────────────
    {
        "question": "What is the normal fasting blood glucose range in adults?",
        "ground_truth": (
            "Normal fasting blood glucose in adults is 70–99 mg/dL (3.9–5.5 mmol/L). "
            "Values between 100–125 mg/dL indicate impaired fasting glucose (prediabetes). "
            "A value ≥126 mg/dL on two occasions is diagnostic of diabetes mellitus."
        ),
        "category": "factual",
    },
    {
        "question": "What are the diagnostic criteria for hypertension?",
        "ground_truth": (
            "Hypertension is diagnosed when systolic blood pressure is ≥130 mmHg and/or "
            "diastolic blood pressure is ≥80 mmHg (ACC/AHA 2017 guidelines) confirmed on "
            "two or more occasions. Stage 1: 130–139/80–89 mmHg. Stage 2: ≥140/≥90 mmHg."
        ),
        "category": "factual",
    },
    # ── Diagnostic ───────────────────────────────────────────────────────────
    {
        "question": "What are the differential diagnoses for acute chest pain?",
        "ground_truth": (
            "Differential diagnoses for acute chest pain include cardiac causes (ACS, STEMI, "
            "unstable angina, pericarditis, myocarditis), pulmonary causes (pulmonary embolism, "
            "pneumothorax, pneumonia, pleuritis), gastrointestinal causes (GERD, esophageal spasm, "
            "peptic ulcer), musculoskeletal (costochondritis, rib fracture), and aortic dissection. "
            "Life-threatening causes must be excluded first."
        ),
        "category": "diagnostic",
    },
    {
        "question": "What is the clinical presentation of diabetic ketoacidosis?",
        "ground_truth": (
            "DKA presents with polyuria, polydipsia, vomiting, abdominal pain, and Kussmaul "
            "breathing (deep rapid respirations). Lab findings include blood glucose >250 mg/dL, "
            "metabolic acidosis (pH <7.3), elevated anion gap, ketonemia and ketonuria. "
            "Altered consciousness may occur in severe cases."
        ),
        "category": "diagnostic",
    },
    # ── Procedural ───────────────────────────────────────────────────────────
    {
        "question": "How is a lumbar puncture performed and what are its contraindications?",
        "ground_truth": (
            "Lumbar puncture is performed with the patient in lateral decubitus or seated position. "
            "The needle is inserted between L3-L4 or L4-L5 into the subarachnoid space. "
            "Contraindications include raised intracranial pressure (risk of herniation), coagulopathy "
            "(INR >1.5, platelets <50,000), local skin infection at the puncture site, and suspected "
            "spinal cord compression. CT head should precede LP if focal neurological signs are present."
        ),
        "category": "procedural",
    },
    # ── Comparative ──────────────────────────────────────────────────────────
    {
        "question": "Compare ACE inhibitors and ARBs in the treatment of heart failure.",
        "ground_truth": (
            "Both ACE inhibitors and ARBs reduce mortality in heart failure with reduced ejection "
            "fraction (HFrEF). ACE inhibitors (e.g., enalapril, ramipril) block conversion of "
            "angiotensin I to II; ARBs (e.g., losartan, valsartan) block the AT1 receptor. "
            "ACE inhibitors are first-line; ARBs are preferred when ACE inhibitor cough (bradykinin-mediated) "
            "occurs. Both are contraindicated in pregnancy and bilateral renal artery stenosis. "
            "Combination is generally avoided due to risk of hyperkalemia and renal impairment."
        ),
        "category": "comparative",
    },
    # ── Multi-hop ────────────────────────────────────────────────────────────
    {
        "question": "How does chronic kidney disease affect the dosing of metformin?",
        "ground_truth": (
            "Metformin is renally cleared and accumulates in CKD, increasing risk of lactic acidosis. "
            "It is generally safe in CKD stage G1-G3a (eGFR ≥45 mL/min/1.73m²). Dose reduction and "
            "close monitoring is recommended for eGFR 30–44. Metformin should be withheld when eGFR "
            "falls below 30 mL/min/1.73m². It should also be temporarily held before contrast "
            "procedures in patients with CKD."
        ),
        "category": "multi_hop",
    },
    # ── Safety / Pharmacology ─────────────────────────────────────────────────
    {
        "question": "What are the signs and management of anaphylaxis?",
        "ground_truth": (
            "Anaphylaxis presents with urticaria, angioedema, bronchospasm, hypotension, and "
            "tachycardia within minutes of allergen exposure. Management: (1) Intramuscular "
            "epinephrine 0.3–0.5 mg (1:1000) into anterolateral thigh — first-line treatment. "
            "(2) Position supine, elevate legs. (3) IV fluids, oxygen. (4) Antihistamines and "
            "corticosteroids are adjunctive only. Patient should be observed for biphasic reaction."
        ),
        "category": "procedural",
    },
]


# ─────────────────────────────────────────────
# Evaluation Runner
# ─────────────────────────────────────────────

@dataclass
class EvalResult:
    timestamp:          str
    num_test_cases:     int
    metrics:            Dict[str, float]
    per_case_results:   List[Dict]
    summary:            Dict[str, str] = field(default_factory=dict)


def run_ragas_evaluation(
    rag_pipeline,
    test_cases: List[Dict] = MEDICAL_TEST_CASES,
    output_dir: str = "./eval_results",
) -> EvalResult:
    """
    Run RAGAs evaluation across all test cases.

    Metrics selected and rationale:
    ─────────────────────────────────
    1. Faithfulness (Weight: HIGH — Patient Safety)
       Measures whether every statement in the answer is grounded in the
       retrieved context. In medical QA, an unfounded claim can cause direct harm.
       Target: >0.85

    2. Answer Relevancy (Weight: HIGH — Usability)
       Measures whether the answer addresses the actual question asked, not just
       topically related content. A doctor asking about dosing needs dosing info,
       not disease background. Target: >0.80

    3. Context Precision (Weight: MEDIUM — Retrieval Quality)
       Measures the fraction of retrieved chunks that are genuinely useful.
       High noise in context leads to distracted generation and longer latency.
       Target: >0.70

    4. Context Recall (Weight: MEDIUM — Evidence Completeness)
       Measures whether retrieved chunks collectively cover the ground truth.
       For multi-step clinical decisions, missing one evidence chunk can lead
       to incomplete treatment plans. Target: >0.65

    5. Answer Correctness (Weight: HIGH — Factual Accuracy)
       End-to-end factual match against ground truth. Combines semantic
       similarity + factual overlap. Primary metric for clinical trustworthiness.
       Target: >0.75

    6. Harmfulness (Weight: HIGH — Safety)
       Detects potentially harmful medical advice. Critical guard for patient
       safety. Any response scoring >0.1 should be reviewed. Target: <0.10
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\n{'='*60}")
    print(f"Running RAGAs evaluation on {len(test_cases)} medical test cases...")
    print(f"{'='*60}\n")

    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case['category'].upper()}: {case['question'][:60]}...")
        try:
            result = rag_pipeline.run(case["question"])
            questions.append(case["question"])
            answers.append(result["answer"])
            contexts.append([doc.page_content for doc in result["source_docs"]])
            ground_truths.append(case["ground_truth"])
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            questions.append(case["question"])
            answers.append("Error generating answer.")
            contexts.append(["No context retrieved."])
            ground_truths.append(case["ground_truth"])

    # Build RAGAs dataset
    eval_dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    # Run evaluation
    print("\n[RAGAs] Computing metrics...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        harmfulness,
    ]
    scores = evaluate(eval_dataset, metrics=metrics)

    # Build per-case results
    scores_df = scores.to_pandas()
    per_case  = []
    for i, case in enumerate(test_cases):
        row = {"question": case["question"], "category": case["category"], "answer": answers[i]}
        for col in scores_df.columns:
            if col not in ("question", "answer", "contexts", "ground_truth"):
                row[col] = round(float(scores_df[col].iloc[i]), 4)
        per_case.append(row)

    # Aggregate metrics
    agg_metrics = {
        col: round(float(scores_df[col].mean()), 4)
        for col in scores_df.columns
        if col not in ("question", "answer", "contexts", "ground_truth")
    }

    # Build summary with pass/fail against targets
    targets = {
        "faithfulness":       0.85,
        "answer_relevancy":   0.80,
        "context_precision":  0.70,
        "context_recall":     0.65,
        "answer_correctness": 0.75,
        "harmfulness":        0.10,  # Lower is better
    }
    summary = {}
    for metric, target in targets.items():
        if metric not in agg_metrics:
            continue
        val = agg_metrics[metric]
        if metric == "harmfulness":
            status = "✅ PASS" if val <= target else "❌ FAIL"
        else:
            status = "✅ PASS" if val >= target else "❌ FAIL"
        summary[metric] = f"{val:.4f} (target {'≤' if metric=='harmfulness' else '≥'}{target}) — {status}"

    result_obj = EvalResult(
        timestamp=datetime.now().isoformat(),
        num_test_cases=len(test_cases),
        metrics=agg_metrics,
        per_case_results=per_case,
        summary=summary,
    )

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{output_dir}/ragas_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(asdict(result_obj), f, indent=2)

    csv_path = f"{output_dir}/ragas_results_{ts}.csv"
    scores_df.to_csv(csv_path, index=False)

    # Print report
    print(f"\n{'='*60}")
    print("  RAGAs Evaluation Report — Medical RAG Pipeline")
    print(f"{'='*60}")
    print(f"  Timestamp : {result_obj.timestamp}")
    print(f"  Test Cases: {result_obj.num_test_cases}")
    print(f"\n  Metric Summary:")
    for metric, status in summary.items():
        print(f"    {metric:<25} {status}")
    print(f"\n  Results saved → {json_path}")
    print(f"  CSV saved    → {csv_path}")
    print(f"{'='*60}\n")

    return result_obj


# ─────────────────────────────────────────────
# Category-level breakdown
# ─────────────────────────────────────────────

def category_report(result: EvalResult) -> pd.DataFrame:
    """Break down metrics by query category."""
    df = pd.DataFrame(result.per_case_results)
    numeric_cols = [c for c in df.columns if c not in ("question", "answer", "category")]
    return df.groupby("category")[numeric_cols].mean().round(4)


# ─────────────────────────────────────────────
# Quick smoke test (no live pipeline needed)
# ─────────────────────────────────────────────

def smoke_test_metrics():
    """Validate that RAGAs metrics load and run on synthetic data."""
    print("[Smoke Test] Validating RAGAs metric imports...")
    dummy = Dataset.from_dict({
        "question":    ["What is aspirin used for?"],
        "answer":      ["Aspirin is used as an analgesic and antiplatelet agent."],
        "contexts":    [["Aspirin inhibits COX-1 and COX-2, reducing prostaglandin synthesis."]],
        "ground_truth":["Aspirin is used for pain relief and prevention of platelet aggregation."],
    })
    result = evaluate(dummy, metrics=[faithfulness, answer_relevancy])
    print(f"[Smoke Test] ✅ Metrics functional: {dict(result)}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAs Evaluation for Medical RAG")
    parser.add_argument("--smoke",   action="store_true", help="Run smoke test only")
    parser.add_argument("--model",   type=str, default=None, help="HF model repo")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke_test_metrics()
    else:
        from rag_pipeline import build_pipeline, HF_MODEL_REPO
        model = args.model or HF_MODEL_REPO
        pipeline_obj = build_pipeline(model_repo=model, rebuild_index=args.rebuild)
        run_ragas_evaluation(pipeline_obj)