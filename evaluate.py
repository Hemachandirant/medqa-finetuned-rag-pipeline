"""
RAGAs Evaluation Suite — MedQA DeepSeek RAG Pipeline
======================================================
Uses the vibrantlabsai/ragas v0.4 API (SingleTurnSample + single_turn_ascore).

Install:
  pip install git+https://github.com/vibrantlabsai/ragas

Metrics:
  1. Faithfulness                        — Is the answer grounded in the retrieved context?
  2. ResponseRelevancy                   — Does the answer address the question?
  3. LLMContextPrecisionWithoutReference — Are retrieved chunks relevant?
  4. LLMContextRecall                    — Does context cover the ground truth answer?

Usage:
  python ragas_eval.py                          # Full eval on built-in test set
  python ragas_eval.py --questions custom.json  # Custom questions file
  python ragas_eval.py --output results.json    # Save results to file
  python ragas_eval.py --sample 3               # Quick test on first N questions
"""

import os
import json
import asyncio
import argparse
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── RAGAs v0.4 imports ────────────────────────────────────────────────────────
from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ── Pipeline ──────────────────────────────────────────────────────────────────
from rag_pipeline import build_pipeline, MedicalRAGPipeline, SAGEMAKER_ENDPOINT, AWS_REGION


# ─────────────────────────────────────────────
# Judge LLM — Azure OpenAI
# RAGAs needs a capable judge. The fine-tuned
# 1.5B SageMaker model cannot follow RAGAs'
# structured scoring prompts reliably.
# ─────────────────────────────────────────────

def get_judge_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT", ""),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
        api_key=os.getenv("AZURE_API_KEY", ""),
        api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
        temperature=0.0,
    ))


def get_judge_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_ENDPOINT", ""),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT", "text-embedding-3-small"),
        api_key=os.getenv("AZURE_API_KEY", ""),
        api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
        chunk_size=10,
    ))


# ─────────────────────────────────────────────
# Built-in Medical Test Set
# Covers all 5 query types + diverse topics
# ─────────────────────────────────────────────

MEDICAL_TEST_SET = [
    # ── FACTUAL ──────────────────────────────
    {
        "question": "What is the normal range of HbA1c for a diabetic patient under control?",
        "ground_truth": "HbA1c below 7% is considered good glycemic control for most diabetic patients according to ADA guidelines.",
        "query_type": "factual",
    },
    {
        "question": "What is metformin and how does it work?",
        "ground_truth": "Metformin is a biguanide that reduces hepatic glucose production and improves peripheral glucose utilization by activating AMP-dependent protein kinase.",
        "query_type": "factual",
    },
    {
        "question": "What is the definition of type 2 diabetes mellitus?",
        "ground_truth": "Type 2 diabetes mellitus is a progressive metabolic disorder characterized by insulin resistance and relative insulin deficiency, leading to chronic hyperglycemia.",
        "query_type": "factual",
    },
    # ── PROCEDURAL ───────────────────────────
    {
        "question": "How is insulin therapy initiated in type 2 diabetes?",
        "ground_truth": "Insulin therapy in type 2 diabetes is typically initiated with basal insulin such as glargine or detemir, titrated based on fasting glucose levels, and combined with oral agents if needed.",
        "query_type": "procedural",
    },
    {
        "question": "How do you perform a diagnostic workup for suspected hypothyroidism?",
        "ground_truth": "Hypothyroidism workup begins with serum TSH measurement. Elevated TSH with low free T4 confirms primary hypothyroidism. Anti-TPO antibodies confirm autoimmune Hashimoto's thyroiditis.",
        "query_type": "procedural",
    },
    # ── DIAGNOSTIC ───────────────────────────
    {
        "question": "What are the symptoms and diagnostic criteria for diabetic ketoacidosis?",
        "ground_truth": "DKA presents with polyuria, polydipsia, nausea, vomiting, and abdominal pain. Criteria: glucose >250 mg/dL, pH <7.3, bicarbonate <18, and positive ketones.",
        "query_type": "diagnostic",
    },
    {
        "question": "What are the differential diagnoses for chest pain with dyspnea?",
        "ground_truth": "Differentials include acute MI, pulmonary embolism, pneumothorax, aortic dissection, pericarditis. ECG, troponin, D-dimer, and chest X-ray guide workup.",
        "query_type": "diagnostic",
    },
    {
        "question": "How do you diagnose chronic kidney disease?",
        "ground_truth": "CKD is diagnosed by persistent GFR below 60 mL/min/1.73m² or markers of kidney damage such as albuminuria for more than 3 months.",
        "query_type": "diagnostic",
    },
    # ── COMPARATIVE ──────────────────────────
    {
        "question": "Compare GLP-1 receptor agonists and SGLT2 inhibitors in type 2 diabetes.",
        "ground_truth": "GLP-1 agonists promote insulin secretion and reduce appetite causing weight loss. SGLT2 inhibitors reduce renal glucose reabsorption and have proven heart failure and renal protective benefits.",
        "query_type": "comparative",
    },
    {
        "question": "What is the difference between type 1 and type 2 diabetes mellitus?",
        "ground_truth": "Type 1 DM is autoimmune beta-cell destruction causing absolute insulin deficiency requiring insulin. Type 2 involves insulin resistance managed initially with lifestyle changes and oral agents.",
        "query_type": "comparative",
    },
    # ── MULTI-HOP ────────────────────────────
    {
        "question": "How does obesity affect insulin resistance and cardiovascular risk?",
        "ground_truth": "Obesity causes adipose dysfunction with elevated free fatty acids and cytokines impairing insulin signaling, promoting dyslipidemia, hypertension, and atherosclerosis.",
        "query_type": "multi_hop",
    },
    {
        "question": "What is the relationship between hypertension and chronic kidney disease progression?",
        "ground_truth": "Hypertension accelerates CKD by increasing glomerular pressure causing nephrosclerosis. ACE inhibitors and ARBs reduce proteinuria and slow CKD progression.",
        "query_type": "multi_hop",
    },
]


# ─────────────────────────────────────────────
# Run pipeline on test set
# ─────────────────────────────────────────────

def run_pipeline_on_testset(
    pipeline: MedicalRAGPipeline,
    test_set: List[Dict],
) -> List[Dict]:
    results = []

    print(f"\n{'='*60}")
    print(f"Running pipeline on {len(test_set)} questions...")
    print(f"{'='*60}\n")

    for i, item in enumerate(test_set, 1):
        q = item["question"]
        print(f"[{i}/{len(test_set)}] {q[:75]}...")

        t0 = time.time()
        try:
            result  = pipeline.run(q)
            elapsed = round(time.time() - t0, 1)

            # Strip "--- Sources ---" section — RAGAs needs clean LLM answer only
            clean_answer = result["answer"].split("--- Sources ---")[0].strip()

            results.append({
                "question":     q,
                "answer":       clean_answer,
                "contexts":     [doc.page_content for doc in result["source_docs"]],
                "ground_truth": item["ground_truth"],
                "query_type":   item.get("query_type", "unknown"),
            })
            print(f"    ✅ {elapsed}s | docs={result['num_docs']} | type={result['query_type']}")

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results.append({
                "question":     q,
                "answer":       "Error: pipeline failed",
                "contexts":     [""],
                "ground_truth": item["ground_truth"],
                "query_type":   item.get("query_type", "unknown"),
            })

    return results


# ─────────────────────────────────────────────
# RAGAs v0.4 scoring — per sample
# ─────────────────────────────────────────────

async def score_sample(item: Dict, metrics: Dict) -> Dict:
    """Score a single item across all 4 metrics using v0.4 API."""
    sample = SingleTurnSample(
        user_input=item["question"],
        response=item["answer"],
        retrieved_contexts=item["contexts"],
        reference=item["ground_truth"],
    )

    scores = {}
    for name, metric in metrics.items():
        try:
            result      = await metric.single_turn_ascore(sample)
            scores[name] = round(float(result), 4) if result is not None else None
        except Exception as e:
            print(f"    ⚠️  {name} failed: {e}")
            scores[name] = None

    return scores


async def run_ragas_evaluation(pipeline_results: List[Dict]) -> List[Dict]:
    print(f"\n{'='*60}")
    print("Running RAGAs evaluation (vibrantlabsai/ragas v0.4)...")
    print(f"{'='*60}\n")

    judge_llm        = get_judge_llm()
    judge_embeddings = get_judge_embeddings()

    metrics = {
        "faithfulness":      Faithfulness(llm=judge_llm),
        "response_relevancy": ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": LLMContextPrecisionWithoutReference(llm=judge_llm),
        "context_recall":    LLMContextRecall(llm=judge_llm),
    }

    scored = []
    for i, item in enumerate(pipeline_results, 1):
        print(f"[{i}/{len(pipeline_results)}] Scoring: {item['question'][:60]}...")
        scores = await score_sample(item, metrics)
        scored.append({**item, **scores})
        print(f"    faith={scores.get('faithfulness')} | "
              f"rel={scores.get('response_relevancy')} | "
              f"prec={scores.get('context_precision')} | "
              f"rec={scores.get('context_recall')}")

    return scored


# ─────────────────────────────────────────────
# Results Formatter
# ─────────────────────────────────────────────

METRICS = ["faithfulness", "response_relevancy", "context_precision", "context_recall"]


def safe_avg(values):
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def print_results(scored: List[Dict]):
    print(f"\n{'='*60}")
    print("RAGAs Evaluation Results")
    print(f"{'='*60}\n")

    print("Overall Scores:")
    print("-" * 45)
    for m in METRICS:
        avg = safe_avg([r.get(m) for r in scored])
        if avg is not None:
            bar  = "█" * int(avg * 20)
            flag = "✅" if avg >= 0.7 else ("⚠️ " if avg >= 0.4 else "❌")
            print(f"  {flag} {m:<30} {avg:.3f}  {bar}")

    print(f"\n{'='*60}")
    print("Per-Question Breakdown:")
    print("-" * 60)
    for i, r in enumerate(scored, 1):
        print(f"\n[Q{i}] ({r['query_type']}) {r['question'][:65]}...")
        for m in METRICS:
            val  = r.get(m)
            disp = f"{val:.3f}" if val is not None else "n/a"
            flag = ("✅" if val >= 0.7 else ("⚠️ " if val >= 0.4 else "❌")) if val is not None else "⚠️ "
            print(f"    {flag} {m:<30} {disp}")

    print(f"\n{'='*60}")
    print("Scores by Query Type:")
    print("-" * 45)
    for qt in sorted(set(r["query_type"] for r in scored)):
        subset = [r for r in scored if r["query_type"] == qt]
        parts  = [f"{m[:6]}: {safe_avg([r.get(m) for r in subset])}" for m in METRICS]
        print(f"  {qt:<14} {' | '.join(parts)}")

    ranked = sorted([r for r in scored if r.get("faithfulness") is not None],
                    key=lambda x: x["faithfulness"])
    if ranked:
        print(f"\n{'='*60}")
        print("Lowest Faithfulness (top 3 to review):")
        print("-" * 45)
        for r in ranked[:3]:
            print(f"  [{scored.index(r)+1}] {r['question'][:65]}...")
            print(f"       faithfulness={r['faithfulness']}")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────

def save_results(scored: List[Dict], output_path: str):
    output = {
        "summary":       {m: safe_avg([r.get(m) for r in scored]) for m in METRICS},
        "num_questions": len(scored),
        "results":       scored,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[Eval] Results saved → {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAGAs v0.4 Evaluation — MedQA RAG Pipeline")
    parser.add_argument("--questions", type=str, default=None,
        help="Path to custom questions JSON [{question, ground_truth, query_type}]")
    parser.add_argument("--output",    type=str, default="ragas_results.json")
    parser.add_argument("--sample",    type=int, default=None,
        help="Only run on the first N questions (quick test)")
    parser.add_argument("--endpoint",  type=str, default=None,
        help="Override SageMaker endpoint name")
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            test_set = json.load(f)
        print(f"[Eval] Loaded {len(test_set)} questions from {args.questions}")
    else:
        test_set = MEDICAL_TEST_SET
        print(f"[Eval] Using built-in test set ({len(test_set)} questions)")

    if args.sample:
        test_set = test_set[:args.sample]
        print(f"[Eval] Sampling first {args.sample} questions")

    endpoint = args.endpoint or SAGEMAKER_ENDPOINT
    pipeline = build_pipeline(endpoint=endpoint, region=AWS_REGION)

    pipeline_results = run_pipeline_on_testset(pipeline, test_set)
    scored           = asyncio.run(run_ragas_evaluation(pipeline_results))

    print_results(scored)
    save_results(scored, args.output)


if __name__ == "__main__":
    main()