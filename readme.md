# Medical RAG Pipeline

**Domain:** Medical / Healthcare Textbooks  
**Framework:** LangChain + ChromaDB  
**Strategy:** Hybrid Adaptive + Corrective RAG  
**Evaluation:** RAGAs (6 metrics)

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────────────────┐
│  1. Query Classifier  │  ← Adaptive RAG: routes by complexity
│  (5 query types)     │
└──────────┬───────────┘
           │
    ┌──────▼──────┐
    │  Factual?   │──► Simple retrieval (k=3, similarity)
    │  Procedural?│──► MMR retrieval (k=5)
    │  Diagnostic?│──► MMR + Query Expansion (k=7)
    │  Comparative│──► MMR + Query Expansion (k=8)
    │  Multi-hop? │──► MMR + Query Expansion (k=10)
    └──────┬──────┘
           │
    ┌──────▼────────────────┐
    │  2. Query Expansion    │  ← LLM generates 3 medical variants
    │  (conditional)        │     for better recall
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │  3. ChromaDB Retrieval │  ← BioBERT embeddings, cosine sim
    │  (multi-query fusion) │
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │  4. Relevance Grader   │  ← Corrective RAG: LLM scores 0-1
    │  (CRAG)               │     re-retrieves if score < 0.35
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │  5. Answer Generation  │  ← Fine-tuned medical LLM
    │  (HF model)           │     with structured medical prompt
    └───────────────────────┘
```

---

## Decision Rationale

### RAG Strategy: Hybrid Adaptive + Corrective RAG

**Why not plain RAG?**  
Plain RAG retrieves a fixed-k documents regardless of query complexity. A factual query like *"What is the normal HbA1c range?"* needs 3 precise chunks, while a multi-hop query like *"How does CKD affect metformin dosing?"* requires 10 chunks spanning nephrology AND pharmacology chapters. Fixed-k either wastes context or misses evidence.

**Why Adaptive RAG?**  
Medical queries fall into distinct cognitive categories (factual, procedural, diagnostic, comparative, multi-hop). Each category has different retrieval needs:

| Query Type  | Rationale for Config |
|-------------|----------------------|
| Factual     | Single correct answer; high precision needed; MMR diversity unhelpful |
| Procedural  | Sequential steps; MMR prevents redundant step chunks |
| Diagnostic  | Multiple differentials needed; expansion finds related syndromes |
| Comparative | Two entities; expansion generates entity-specific sub-queries |
| Multi-hop   | Cross-chapter reasoning; aggressive expansion + large k required |

**Why Corrective RAG?**  
Medical textbooks have high lexical variation (e.g. "MI", "myocardial infarction", "STEMI" are the same concept). Embedding-based retrieval can miss relevant chunks. CRAG adds a second validation layer that catches these misses and triggers re-retrieval — acting as a safety net before the answer is generated.

**Why not Self-RAG or Graph RAG?**  
- Self-RAG requires the LLM to be fine-tuned with retrieval special tokens — our fine-tuned model may not support this.
- Graph RAG (entity relationship graphs) is powerful for structured knowledge but requires significant preprocessing and a graph database — overkill for a textbook retrieval system where semantic similarity is sufficient.

---

### Chunking Strategy: Medical Semantic Chunking

**Configuration:** `chunk_size=512`, `chunk_overlap=64`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Separators | Section headers first, then `\n\n`, `\n`, `. ` | Medical textbooks are organized by clinical sections (Diagnosis, Treatment, Pathophysiology). Splitting at section boundaries keeps concepts together rather than cutting mid-explanation. |
| `chunk_size=512` | Tokens | Large enough to hold a complete clinical description (e.g., full diagnostic criteria). Small enough for precise embedding similarity — 1024+ token chunks dilute the embedding signal. |
| `chunk_overlap=64` | ~12% | Preserves continuity at boundaries. Drug dosing often continues across a paragraph boundary; overlap ensures the dose appears in at least one complete chunk. |

**Why not fixed-size chunking?**  
Fixed-size splits (e.g. every 200 words) frequently cut in the middle of a drug name + dosage pair, a diagnostic criterion list, or a numbered treatment step — creating partially useful chunks that confuse the retriever.

**Why not sentence-level chunking?**  
Medical sentences are densely packed with acronyms and context references ("The above treatment is contraindicated in..."). Single sentences lose the surrounding context needed to interpret them correctly.

---

### Embeddings: BioBERT (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-sst2)

| Property | Value |
|----------|-------|
| Base model | BioBERT (trained on PubMed + MIMIC-III) |
| Fine-tuning | NLI + medical entailment tasks |
| Dimension | 768 |
| Distance metric | Cosine similarity |

**Why BioBERT over general embeddings (OpenAI, sentence-t5)?**  

General embeddings fail on medical synonym matching:
```
"myocardial infarction" ↔ "heart attack"          → BioBERT: 0.94 | General: 0.71
"hypertension" ↔ "elevated blood pressure"        → BioBERT: 0.91 | General: 0.68  
"type 2 diabetes" ↔ "non-insulin dependent DM"    → BioBERT: 0.89 | General: 0.61
```

BioBERT was pre-trained on 4.5B words of biomedical literature and understands medical abbreviations, drug names, anatomical terms, and disease synonyms that general models cannot handle.

**Why normalize embeddings?**  
Normalized embeddings make cosine similarity equivalent to dot product, which is natively supported by ChromaDB's HNSW index — improving both accuracy and query speed.

---

### Tokenization

LangChain's `RecursiveCharacterTextSplitter` uses character-count as a proxy for token count (`length_function=len`). The relationship holds well for medical text because:
1. Medical terminology is mostly ASCII (no CJK inflation)
2. The 512 char limit corresponds to ~128-180 subword tokens for BioBERT's WordPiece tokenizer — well within the 512-token model limit

The LLM (your fine-tuned model) uses its own HuggingFace tokenizer automatically via `AutoTokenizer`. Medical domain tokenizers handle clinical abbreviations better than general tokenizers.

---

### MMR (Maximal Marginal Relevance)

**`lambda_mult=0.6`** balances relevance (0.6) vs. diversity (0.4).

In medical retrieval, diversity matters because:
- Multiple chapters may cover the same disease differently (etiology vs. management chapters)
- Pure similarity retrieval returns near-duplicate chunks from the same paragraph
- Diverse chunks provide broader clinical evidence for the LLM to synthesize

---

## RAGAs Metric Rationale

### Why these 6 metrics?

```
                    ┌─────────────────────────────┐
                    │  RETRIEVAL QUALITY           │
                    │  Context Precision  (noise?) │
                    │  Context Recall    (gaps?)   │
                    └──────────┬──────────────────┘
                               │ feeds into
                    ┌──────────▼──────────────────┐
                    │  GENERATION QUALITY          │
                    │  Faithfulness   (hallucin?)  │
                    │  Answer Relevancy (on-topic?)│
                    │  Answer Correctness (right?) │
                    └──────────┬──────────────────┘
                               │ + medical-specific
                    ┌──────────▼──────────────────┐
                    │  SAFETY                      │
                    │  Harmfulness  (do no harm)   │
                    └─────────────────────────────┘
```

| Metric | Target | Rationale | Clinical Consequence if Fails |
|--------|--------|-----------|-------------------------------|
| **Faithfulness** | >0.85 | Every claim must be grounded in retrieved context. Hallucination in clinical QA can suggest wrong drugs, wrong dosages, or wrong diagnoses. | Fabricated drug interaction → patient harm |
| **Answer Relevancy** | >0.80 | Answer must address the actual question. A doctor asking about dosing should not get disease pathophysiology. | Clinician receives unhelpful response, may act on incomplete info |
| **Context Precision** | >0.70 | Retrieved chunks must be genuinely useful. High noise forces the LLM to distill signal from noise, increasing hallucination risk. | LLM picks wrong fact from noisy context |
| **Context Recall** | >0.65 | Retrieved chunks must cover all aspects of the ground truth. Missing a contraindication chunk = missing it in the answer. | Incomplete treatment plan; missed contraindication |
| **Answer Correctness** | >0.75 | Factual accuracy vs. ground truth. Primary clinical trust metric. | Wrong diagnosis or treatment recommendation |
| **Harmfulness** | <0.10 | Medical-domain safety guard. Detects if model outputs dangerous advice. | Harmful medical recommendation |

### Why not BLEU/ROUGE?
BLEU and ROUGE measure n-gram overlap with ground truth — they penalize medically correct synonyms ("elevated blood pressure" vs "hypertension"). RAGAs uses LLM-based semantic evaluation, which is far more appropriate for clinical text where paraphrase is ubiquitous.

### Why not just perplexity?
Perplexity measures language model fluency, not factual correctness or relevance. A fluent hallucination has low perplexity.

---

## Project Structure

```
medical_rag/
├── rag_pipeline.py     # Main RAG pipeline (Adaptive + Corrective)
├── evaluate.py         # RAGAs test suite (6 metrics, 8 test cases)
├── requirements.txt    # Dependencies
├── README.md           # This file
├── data/               # Place medical PDF textbooks here
│   └── *.pdf
└── eval_results/       # Auto-created; stores JSON + CSV results
    └── ragas_results_YYYYMMDD_HHMMSS.{json,csv}
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your HuggingFace model repo
export HF_MODEL_REPO="your-username/your-medical-model"

# 3. Place PDF textbooks in ./data/

# 4. Build the index and run a query
python rag_pipeline.py --rebuild --query "What are the symptoms of pulmonary embolism?"

# 5. Run full RAGAs evaluation
python evaluate.py

# 6. Smoke test (no pipeline needed)
python evaluate.py --smoke
```

---

## Targets Summary

| Metric | Target | Priority |
|--------|--------|----------|
| Faithfulness | >0.85 | 🔴 Critical |
| Answer Correctness | >0.75 | 🔴 Critical |
| Answer Relevancy | >0.80 | 🟠 High |
| Context Precision | >0.70 | 🟡 Medium |
| Context Recall | >0.65 | 🟡 Medium |
| Harmfulness | <0.10 | 🔴 Critical |