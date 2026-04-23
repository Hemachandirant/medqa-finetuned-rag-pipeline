# MedQA — Fine-tuned DeepSeek RAG Pipeline

**Model:** DeepSeek-R1-Distill-Qwen-1.5B (fine-tuned on MedQA)  
**RAG Strategy:** Hybrid Adaptive + Corrective RAG  
**Vector Store:** Pinecone (cloud-hosted, serverless)  
**Embeddings:** Azure OpenAI `text-embedding-3-small`  
**Deployment:** AWS SageMaker (`ap-south-1`, endpoint: `medqa-deepseek-v2`)  
**API:** FastAPI — `POST /query`, `GET /health`

---

## Architecture

```
User Query (POST /query)
        │
        ▼
┌─────────────────────────┐
│  1. Query Classifier     │  Adaptive RAG: classifies into 5 types
│  (regex pattern match)  │  factual / procedural / diagnostic /
└──────────┬──────────────┘  comparative / multi_hop
           │
    ┌──────▼──────────────────────────────┐
    │  2. Retrieval Config (per type)      │
    │  factual    → k=5,  mmr=off          │
    │  procedural → k=5,  mmr=on           │
    │  diagnostic → k=7,  mmr=on, expand   │
    │  comparative→ k=8,  mmr=on, expand   │
    │  multi_hop  → k=10, mmr=on, expand   │
    └──────┬──────────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │  3. Query Expansion (if enabled)     │  LLM generates 3 medical
    │  SageMaker → 3 query variants        │  variants; retrieval runs
    └──────┬──────────────────────────────┘  on all 4, deduplicated
           │
    ┌──────▼──────────────────────────────┐
    │  4. Pinecone Retrieval               │  Azure text-embedding-3-small
    │  cosine similarity / MMR             │  11,593 indexed chunks
    │  lambda_mult=0.6, fetch_k=k×3        │  from medical textbooks
    └──────┬──────────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │  5. Corrective RAG — Grader          │  Keyword overlap score
    │  threshold=0.35                      │  re-retrieves if < 2 docs
    │  re-retrieve at 0.7× threshold       │  pass (no LLM call needed)
    └──────┬──────────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │  6. Answer Generation                │  DeepSeek-R1-Distill-Qwen-1.5B
    │  SageMaker endpoint invocation       │  context capped at 6000 chars
    │  + source attribution appended       │  <think> tokens stripped
    └─────────────────────────────────────┘
```

---

## Project Structure

```
medqa-finetuned-rag-pipeline/
├── main.py               # FastAPI app (POST /query, GET /health)
├── rag_pipeline.py       # Full RAG pipeline (Adaptive + Corrective)
├── index_documents.py    # Pinecone indexing from .txt textbook files
├── evaluate.py           # RAGAs v0.4 evaluation suite (4 metrics, 12 questions)
├── ragas_results.json    # Latest evaluation output
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── render.yaml           # Render deployment config
├── Procfile              # Gunicorn startup for production
├── data/                 # Medical textbook .txt files go here
└── model deployment/     # SageMaker deployment scripts
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Hemachandirant/medqa-finetuned-rag-pipeline
cd medqa-finetuned-rag-pipeline
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Fill in SAGEMAKER_ENDPOINT_NAME, AWS credentials, PINECONE_API_KEY, AZURE keys

# 3. Place medical textbook .txt files in ./data/

# 4. Build Pinecone index
python index_documents.py

# 5. Start the API
uvicorn main:app --host 0.0.0.0 --port 8000

# 6. Query the API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the treatment options for type 2 diabetes?"}'

# 7. Run RAGAs evaluation
python evaluate.py

# 8. Quick smoke test (first 3 questions)
python evaluate.py --sample 3
```

---

## Environment Variables

```bash
# AWS SageMaker
SAGEMAKER_ENDPOINT_NAME=medqa-deepseek-v2
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=medqa-textbooks
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Azure OpenAI (embeddings + RAGAs judge)
AZURE_ENDPOINT=...
AZURE_API_KEY=...
AZURE_API_VERSION=2024-02-01
AZURE_DEPLOYMENT=text-embedding-3-small
AZURE_CHAT_DEPLOYMENT=gpt-4o
```

---

## API Reference

### `POST /query`

**Request:**
```json
{ "query": "What are the symptoms and diagnostic criteria for DKA?" }
```

**Response:**
```json
{
  "query": "What are the symptoms...",
  "query_type": "diagnostic",
  "answer": "DKA presents with...\n\n--- Sources ---\n[Source 1 — Harrison | relevance=0.72]...",
  "num_docs": 5,
  "source_documents": [
    { "book": "InternalMed_Harrison", "relevance_score": 0.72, "section_type": "diagnosis", "excerpt": "..." }
  ],
  "raw_context": "...",
  "latency_ms": 3420.5
}
```

### `GET /health`

```json
{ "status": "ok", "pipeline_ready": true, "endpoint": "medqa-deepseek-v2", "region": "ap-south-1" }
```

Swagger UI is auto-generated at `/docs`.

---

## Decision Rationale

### 1. Model: DeepSeek-R1-Distill-Qwen-1.5B

The 1.5B distilled model was chosen over larger variants for three reasons. First, it fits within a SageMaker `ml.g4dn.xlarge` instance (T4, 16GB VRAM) without quantization, keeping deployment within free-tier credit budgets. Second, DeepSeek-R1 distillation transfers chain-of-thought reasoning from the much larger R1 teacher model — this matters for multi-step clinical questions where intermediate reasoning affects answer quality. Third, the MedQA fine-tuning dataset is small enough that a 1.5B model trains in under 2 hours on a T4; larger models would require significantly more resources for diminishing marginal gains on this dataset size.

**Chat template:** The `<|im_start|>/<|im_end|>` ChatML format is used because DeepSeek-R1-Distill-Qwen is built on Qwen2, which was pre-trained with this template. Using the wrong template causes the model to ignore the system prompt entirely.

**`<think>` stripping:** DeepSeek-R1 emits chain-of-thought reasoning inside `<think>...</think>` tags before the final answer. These are stripped in `SageMakerLLM._call()` with a regex so the API returns only the clean answer.

**Temperature = 0.2:** Medical QA is a factual, low-variance task. Low temperature keeps the model anchored to retrieved context. `repetition_penalty=1.05` prevents the model from looping on medical terminology, which is a known failure mode for small models on dense clinical text.

---

### 2. RAG Strategy: Hybrid Adaptive + Corrective RAG

**Why not plain RAG?**

Plain RAG retrieves a fixed number of documents regardless of query complexity. A factual lookup (`"What is the normal HbA1c range?"`) needs 3 precise chunks. A multi-hop question (`"How does obesity affect insulin resistance and cardiovascular risk?"`) requires evidence from endocrinology, cardiology, and adipose biology chapters simultaneously. Fixed-k either wastes the context window with irrelevant chunks or misses cross-chapter evidence.

**Why Adaptive RAG?**

Medical queries fall into five distinct cognitive categories, each with different evidence needs:

| Query Type | k | MMR | Expand | Rationale |
|---|---|---|---|---|
| Factual | 5 | off | off | Single correct answer; high precision needed; MMR diversity unhelpful |
| Procedural | 5 | on | off | Sequential steps; MMR prevents duplicate step chunks |
| Diagnostic | 7 | on | on | Multiple differentials needed; expansion finds related syndromes |
| Comparative | 8 | on | on | Two drug classes; expansion generates entity-specific sub-queries |
| Multi-hop | 10 | on | on | Cross-chapter reasoning; aggressive expansion required |

The classifier (`classify_query()`) uses regex pattern matching — fast, deterministic, and sufficient for the well-defined vocabulary of clinical question types. No LLM call is needed at the routing step.

**Why Corrective RAG (CRAG)?**

Medical textbooks use highly variable terminology. "Myocardial infarction", "MI", "STEMI", and "acute coronary syndrome" refer to overlapping concepts but use different surface forms. Embedding similarity alone can miss chunks using a different abbreviation. CRAG adds a keyword-overlap grader (`grade_documents()`) that catches these semantic misses and triggers re-retrieval at a relaxed threshold (0.7× original). Importantly, the grader is keyword-based — not LLM-based — to avoid adding a full extra SageMaker round trip (and associated latency + cost) before every answer.

**Why not Self-RAG?**

Self-RAG requires the model to be fine-tuned with special retrieval decision tokens (`[Retrieve]`, `[ISREL]`, `[ISSUP]`). The current fine-tuning dataset is MedQA QA pairs, which do not include these tokens. Retrofitting Self-RAG would require re-fine-tuning with an augmented dataset.

**Why not Graph RAG?**

Graph RAG builds entity-relationship graphs over the corpus and traverses them at query time. It is powerful for structured knowledge (drug-gene-disease networks) but requires significant preprocessing and a graph database. The MedQA textbooks are narrative medical prose — semantic similarity retrieval is the right primary mechanism, and CRAG provides adequate correction of its failures.

---

### 3. Chunking Strategy

**Configuration:** `chunk_size=2500 chars`, `chunk_overlap=200 chars`

```python
MEDICAL_SEPARATORS = [
    "\nDiagnosis", "\nTreatment", "\nPathophysiology",
    "\nClinical Features", "\nManagement", "\nEtiology",
    "\nComplications", "\nDefinition", "\nEpidemiology",
    "\nPrognosis", "\n## ", "\n### ", "\n\n", "\n"
]
```

**Why section-aware separators?**

Harrison's Principles of Internal Medicine and similar textbooks are organized into named clinical sections. Splitting at these boundaries keeps conceptually complete content together. A chunk that starts mid-sentence in a "Treatment" section and ends mid-sentence in a "Complications" section answers neither question well and creates noisy embeddings.

**Why `chunk_size=2500` characters?**

`text-embedding-3-small` accepts up to 8191 tokens. A 2500-character chunk is approximately 600–700 subword tokens — well within model limits, large enough to hold a complete clinical description (e.g. the full DKA diagnostic criteria including glucose threshold, pH cutoff, bicarbonate level, and ketone requirement), and small enough that the embedding vector remains specific rather than averaging over too many concepts.

**Why `chunk_overlap=200`?**

Drug dosing and treatment criteria frequently span paragraph boundaries. A 200-character overlap ensures boundary content (e.g. a drug name in the last sentence of one paragraph and its dose in the first sentence of the next) appears complete in at least one chunk.

**Why not sentence-level chunking?**

Medical sentences are extremely dense with context references. "In the absence of unequivocal hyperglycemia and acute metabolic decompensation, these criteria should be confirmed" only makes sense with the surrounding context that defines "these criteria". Single-sentence chunks systematically strip this context.

**Section tagging:** Each chunk receives a `section_type` metadata field (diagnosis, treatment, pathophysiology, management, etiology, general), surfaced in the API response for source attribution.

---

### 4. Embeddings: Azure OpenAI `text-embedding-3-small`

| Property | Value |
|---|---|
| Provider | Azure OpenAI |
| Model | text-embedding-3-small |
| Dimension | 1536 |
| Distance metric | Cosine (Pinecone default) |
| Batch size | 10 chunks per API call |

**Why `text-embedding-3-small` over BioBERT?**

The practical constraint is the vector store. Pinecone Serverless requires a single fixed embedding dimension for the entire index. `text-embedding-3-small` (1536 dims) is available on Azure OpenAI credits already being used for the project. It was trained with Matryoshka Representation Learning, maintaining high quality across dimension ranges, and it handles the mixed vocabulary of medical textbooks (Latin terms, abbreviations, brand/generic drug names) effectively due to its broad training corpus.

**Why cosine similarity?**

Cosine similarity is length-invariant — a 500-char chunk and a 2500-char chunk discussing the same concept score similarly regardless of their difference in raw token count. This matters because medical textbook sections vary dramatically in length.

---

### 5. Tokenization

LangChain's `RecursiveCharacterTextSplitter` uses `length_function=len` (character count as a proxy for token count). This is appropriate because:

- Medical textbook text is almost entirely ASCII — no CJK character inflation
- 2500 characters corresponds to roughly 600–700 subword tokens for the BPE tokenizers used by both the embedding model and the LLM
- The Azure OpenAI embedding API enforces its own token limit server-side

The LLM tokenizer (`AutoTokenizer`) is handled automatically by SageMaker's HuggingFace inference container — the pipeline sends a fully formatted ChatML string, and the container tokenizes it internally using the model's bundled tokenizer.

---

### 6. MMR — Maximal Marginal Relevance

`lambda_mult=0.6`, `fetch_k = k × 3`

MMR balances relevance (0.6 weight) against diversity (0.4 weight). In medical retrieval, diversity is essential because the same disease is covered across multiple chapters (etiology, management, complications). Pure similarity search returns near-duplicate paragraphs from the same section. MMR surfaces cross-chapter chunks that together give the model a more complete clinical picture. `fetch_k = k × 3` ensures MMR has a large enough candidate pool to make meaningful diversity selections.

---

### 7. RAGAs Evaluation

**Why these four metrics?**

The four metrics cover both layers of a RAG system:

```
Retrieval quality             Generation quality
─────────────────             ──────────────────
Context Precision    →        Faithfulness
Context Recall       →        Response Relevancy
```

Together they answer the key diagnostic question: *did retrieval give the model the right information, and did the model use it correctly?*

| Metric | What it measures | Clinical consequence if it fails | Score |
|---|---|---|---|
| **Faithfulness** | All answer claims are grounded in retrieved context | Hallucinated drug names or dosages are patient safety risks | **0.865** |
| **Response Relevancy** | Answer addresses the actual question asked | Clinician asking about dosing gets pathophysiology instead | **0.681** |
| **Context Precision** | Retrieved chunks are genuinely useful | Noisy context increases the chance the model picks the wrong fact | **0.718** |
| **Context Recall** | Retrieved context covers the ground truth | Missing a contraindication chunk = missing it in the answer | **0.917** |

**Why not BLEU or ROUGE?**

BLEU and ROUGE measure n-gram overlap with a reference string. They penalize medically correct synonyms — "hypertension" and "elevated blood pressure" score near-zero overlap despite being clinically identical. RAGAs uses `gpt-4o` as a semantic judge, which is the correct approach for clinical text where paraphrase is ubiquitous.

**Why not perplexity?**

Perplexity is a fluency metric, not a factual accuracy metric. A confident, fluent hallucination has low perplexity. It tells us nothing about whether the answer is correct or safe.

**Why `gpt-4o` as judge, not the fine-tuned 1.5B model?**

RAGAs requires the judge to follow structured multi-step scoring prompts reliably and produce parseable JSON scores. The fine-tuned 1.5B DeepSeek model is too small to do this consistently. `gpt-4o` is used only for evaluation — all inference uses the fine-tuned SageMaker endpoint.

**Results summary (n=12 questions):**

```
Context Recall     0.917  ████████████████████  excellent
Faithfulness       0.865  █████████████████     strong
Context Precision  0.718  ██████████████        moderate
Response Relevancy 0.681  █████████████         moderate
```

The retriever finds the right passages nearly every time (recall 0.917) but occasionally pulls irrelevant chunks alongside them (precision 0.718). Diagnostic and comparative query types show the weakest precision (0.0 for DKA, 0.0 for GLP-1/SGLT2 comparison) — these require more targeted query expansion. The GLP-1/SGLT2 faithfulness score (0.33) indicates the model confused alpha-glucosidase inhibitor mechanism with GLP-1 mechanism, a fine-tuning gap that more domain-specific training examples would address.

---

## Deployment — AWS SageMaker

The fine-tuned model is deployed as a real-time inference endpoint on SageMaker in `ap-south-1`.

**Endpoint:** `medqa-deepseek-v2`  
**Instance:** `ml.g4dn.xlarge` (1× T4 GPU, 16GB VRAM) — sufficient for 1.5B at float16  
**Invocation:** `boto3` `sagemaker-runtime` client with JSON payload

The FastAPI app loads the full pipeline on startup via `@app.on_event("startup")` so the Pinecone connection and SageMaker client are warm for all subsequent requests. A `GET /health` endpoint confirms readiness before requests are served.

---

## Full Decision Summary

| Component | Choice | Rationale |
|---|---|---|
| Base model | DeepSeek-R1-Distill-Qwen-1.5B | Fits T4 GPU; inherits chain-of-thought from R1 distillation |
| Fine-tuning data | MedQA QA pairs | Teaches clinical answer format with domain-specific QA |
| RAG strategy | Adaptive + Corrective | Query-type routing + keyword-graded relevance validation |
| Vector store | Pinecone Serverless | Cloud-accessible from SageMaker; no infra to manage |
| Embeddings | Azure text-embedding-3-small | Available on existing Azure credits; strong semantic similarity |
| Chunk size | 2500 chars | Holds complete clinical descriptions; ~650 tokens |
| Separators | Clinical section headers | Keeps Diagnosis/Treatment/Pathophysiology sections intact |
| Overlap | 200 chars | Preserves boundary context for dosing and criteria lists |
| MMR lambda | 0.6 | Balances relevance with cross-chapter diversity |
| CRAG threshold | 0.35 | Filters noise without discarding relevant chunks |
| LLM temperature | 0.2 | Factual QA; low variance; repetition_penalty=1.05 |
| RAGAs judge | gpt-4o | 1.5B model cannot follow structured scoring prompts reliably |
| RAGAs metrics | Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall | Cover retrieval + generation; semantically appropriate for medical text |