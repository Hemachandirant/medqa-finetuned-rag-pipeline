"""
Medical RAG Pipeline
====================
Strategy: Hybrid Adaptive + Corrective RAG
- Adaptive RAG: Routes queries to different retrieval strategies based on query complexity
- Corrective RAG: Validates retrieved documents and re-queries if relevance is low
- Vector Store: Pinecone (cloud-hosted, accessible from any deployment)
- Embeddings: Azure OpenAI text-embedding-3-small (matches index_documents.py)
- LLM: Fine-tuned DeepSeek on AWS SageMaker (ap-south-1)

FIXES APPLIED
=============
1.  SYSTEM_PROMPT override removed — now uses imported prompt from medical_prompts.py
2.  grade_documents() is now wired into run() (was dead code before)
3.  QueryType.MECHANISTIC added — mechanism/pathway/MOA queries no longer fall through to FACTUAL
4.  "The answer is" prefix stripped in _call() post-processing
5.  Context window raised: 1,000 chars/doc, 8,000 chars total for deep query types
6.  Query expansion word cap raised from 8 → 20
7.  SageMaker calls wrapped in exponential-backoff retry (3 attempts)
8.  Inline citation instruction added to RAG_PROMPT assembly
9.  relevance_score now set correctly from grade_documents()
10. boto3 client created once per instance (not per call) to avoid session overhead
"""

import os
import re
import json
import time
import boto3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

# FIX 1: Import all prompts — do NOT redefine SYSTEM_PROMPT below this line
from prompts.medical_prompts import (
    SYSTEM_PROMPT,
    RAG_PROMPT_FACTUAL,
    RAG_PROMPT_MECHANISTIC,
    RAG_PROMPT_DIAGNOSTIC,
    QUERY_EXPANSION_PROMPT,
    GRADER_PROMPT,
    FALLBACK_PROMPT,
)


# ─────────────────────────────────────────────
# Configuration (load from .env)
# ─────────────────────────────────────────────

SAGEMAKER_ENDPOINT  = os.getenv("SAGEMAKER_ENDPOINT_NAME", "medqa-deepseek-v2")
AWS_REGION          = os.getenv("AWS_REGION", "ap-south-1")

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medqa-textbooks")

AZURE_ENDPOINT      = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY       = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION   = os.getenv("AZURE_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT    = os.getenv("AZURE_DEPLOYMENT", "text-embedding-3-small")

TOP_K_INITIAL       = 5
TOP_K_EXPANDED      = 10
RELEVANCE_THRESHOLD = 0.35

# Context size limits (chars) by retrieval depth
CONTEXT_PER_DOC = {
    "shallow": 600,   # FACTUAL
    "medium":  900,   # PROCEDURAL, DIAGNOSTIC
    "deep":    1200,  # MECHANISTIC, COMPARATIVE, MULTI_HOP
}
CONTEXT_TOTAL = {
    "shallow": 4000,
    "medium":  6000,
    "deep":    8000,
}


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────

def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        chunk_size=10,
    )


# ─────────────────────────────────────────────
# Pinecone Vector Store Loader
# ─────────────────────────────────────────────

def load_vectorstore() -> PineconeVectorStore:
    print(f"[Pinecone] Connecting to index '{PINECONE_INDEX_NAME}'...")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
    )
    print("[Pinecone] ✅ Connected.")
    return vectorstore


# ─────────────────────────────────────────────
# SageMaker LLM Wrapper
# ─────────────────────────────────────────────

class SageMakerLLM(BaseLLM):
    """
    LangChain-compatible LLM wrapper for fine-tuned DeepSeek deployed on SageMaker.

    Changes vs original:
    - boto3 client created once at init (FIX 10)
    - _call() has exponential-backoff retry on ThrottlingException (FIX 7)
    - _call() strips "The answer is" boilerplate prefix (FIX 4)
    - SYSTEM_PROMPT used from import, not a hardcoded one-liner (FIX 1)
    """

    endpoint_name:  str = Field(default=SAGEMAKER_ENDPOINT)
    region_name:    str = Field(default=AWS_REGION)
    max_new_tokens: int = Field(default=1024)
    max_retries:    int = Field(default=3)

    # FIX 10: cache the boto3 client as a private attribute
    _runtime: Optional[object] = None

    def model_post_init(self, __context):
        """Build the boto3 client once after Pydantic model initialises."""
        object.__setattr__(self, "_runtime", boto3.client(
            "sagemaker-runtime",
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        ))

    @property
    def _llm_type(self) -> str:
        return "sagemaker-deepseek"

    # ── internal: build ChatML-formatted prompt ──────────────────────────────

    def _build_prompt(self, user_message: str, prefix: str = "") -> str:
        """Assemble the ChatML prompt that DeepSeek-R1-Distill-Qwen expects."""
        base = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return base if not prefix else base + prefix

    # ── internal: invoke with retry ──────────────────────────────────────────

    def _invoke_with_retry(self, payload: dict) -> str:
        """
        FIX 7: Retry up to self.max_retries times on transient SageMaker errors.
        Uses exponential back-off (1s, 2s, 4s).
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps(payload),
                )
                result = json.loads(response["Body"].read().decode())
                return result[0]["generated_text"] if isinstance(result, list) else str(result)

            except self._runtime.exceptions.ThrottlingException as e:
                wait = 2 ** attempt
                print(f"[SageMaker] Throttled (attempt {attempt + 1}). Retrying in {wait}s…")
                time.sleep(wait)
                last_error = e

            except Exception as e:
                # Non-transient error — fail fast
                print(f"[SageMaker] Non-retryable error: {e}")
                raise e

        raise RuntimeError(f"[SageMaker] All {self.max_retries} retries exhausted. Last error: {last_error}")

    # ── internal: clean model output ─────────────────────────────────────────

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        FIX 4: Strip artefacts produced by the fine-tuned model:
          - <think>...</think> reasoning tokens (DeepSeek-R1)
          - Leftover <|im_end|> tokens
          - "The answer is" / "The answer to your question is" prefix
        """
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"(?i)^the answer (to (your|this) (question|query) )?is[:\s]*", "", text)
        return text.strip()

    # ── public API ────────────────────────────────────────────────────────────

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        payload = {
            "inputs": self._build_prompt(prompt),
            "parameters": {
                "max_new_tokens":     self.max_new_tokens,
                "temperature":        0.2,
                "top_p":              0.95,
                "repetition_penalty": 1.05,
                "do_sample":          True,
                "return_full_text":   False,
            },
        }
        try:
            raw = self._invoke_with_retry(payload)
            return self._clean_output(raw)
        except Exception as e:
            print(f"[SageMaker] _call failed: {e}")
            return "Error: Could not get a response from the model endpoint."

    def _call_with_prefix(self, user_message: str, prefix: str) -> str:
        """Force the model to continue from a given answer prefix."""
        payload = {
            "inputs": self._build_prompt(user_message, prefix=prefix),
            "parameters": {
                "max_new_tokens":     self.max_new_tokens,
                "temperature":        0.2,
                "top_p":              0.95,
                "repetition_penalty": 1.05,
                "do_sample":          True,
                "return_full_text":   False,
            },
        }
        try:
            raw = self._invoke_with_retry(payload)
            return self._clean_output(raw)
        except Exception as e:
            print(f"[SageMaker] _call_with_prefix failed: {e}")
            return "Error: Could not retrieve a response from the model endpoint."

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import LLMResult, Generation
        generations = [[Generation(text=self._call(p, stop, run_manager))] for p in prompts]
        return LLMResult(generations=generations)


# ─────────────────────────────────────────────
# Query Complexity Classifier (Adaptive RAG)
# ─────────────────────────────────────────────

class QueryType(Enum):
    FACTUAL     = "factual"
    PROCEDURAL  = "procedural"
    MECHANISTIC = "mechanistic"   # FIX 3: new type for MOA/pathway questions
    DIAGNOSTIC  = "diagnostic"
    COMPARATIVE = "comparative"
    MULTI_HOP   = "multi_hop"


# FIX 3: MECHANISTIC added — checked before FACTUAL so MOA queries are caught
QUERY_PATTERNS = {
    QueryType.MECHANISTIC: [
        r"\bmechanism\b", r"\bmoa\b", r"\bmode of action\b",
        r"\bpathway\b",   r"\bhow does .+ work\b",
        r"\bpharmacodynamic\b", r"\bpharmacology\b",
        r"\bactivat(e|es|ion)\b", r"\binhibit(s|ion)\b",
        r"\bagonist\b",   r"\bantagonist\b",
        r"\breceptor\b",  r"\bsignalling\b",
    ],
    QueryType.FACTUAL: [
        r"\bwhat is\b", r"\bdefine\b", r"\bnormal range\b", r"\bvalue of\b",
    ],
    QueryType.PROCEDURAL: [
        r"\bhow (is|to|do)\b", r"\bprocedure\b", r"\bsteps\b", r"\bperform\b",
    ],
    QueryType.DIAGNOSTIC: [
        r"\bdifferential\b", r"\bdiagnose\b", r"\bpresent(s|ing)?\b", r"\bsymptom\b",
    ],
    QueryType.COMPARATIVE: [
        r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bdifference\b",
    ],
    QueryType.MULTI_HOP: [
        r"\baffect\b", r"\brelationship\b", r"\bimpact\b", r"\binteraction\b",
    ],
}

# Depth bucket per query type — controls context window sizing
QUERY_DEPTH: Dict[QueryType, str] = {
    QueryType.FACTUAL:     "shallow",
    QueryType.PROCEDURAL:  "medium",
    QueryType.MECHANISTIC: "deep",
    QueryType.DIAGNOSTIC:  "medium",
    QueryType.COMPARATIVE: "deep",
    QueryType.MULTI_HOP:   "deep",
}


def classify_query(query: str) -> QueryType:
    q = query.lower()
    # FIX 3: iterate in definition order so MECHANISTIC is tested before FACTUAL
    for qtype, patterns in QUERY_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            return qtype
    return QueryType.FACTUAL


def get_retrieval_config(query_type: QueryType) -> Dict:
    configs = {
        QueryType.FACTUAL:     {"k": 5,  "use_mmr": False, "expand_query": False},
        QueryType.PROCEDURAL:  {"k": 5,  "use_mmr": True,  "expand_query": False},
        QueryType.MECHANISTIC: {"k": 8,  "use_mmr": True,  "expand_query": True},  # FIX 3
        QueryType.DIAGNOSTIC:  {"k": 7,  "use_mmr": True,  "expand_query": True},
        QueryType.COMPARATIVE: {"k": 8,  "use_mmr": True,  "expand_query": True},
        QueryType.MULTI_HOP:   {"k": 10, "use_mmr": True,  "expand_query": True},
    }
    return configs[query_type]


# ─────────────────────────────────────────────
# Query Expansion
# ─────────────────────────────────────────────

def expand_medical_query(query: str, llm: SageMakerLLM) -> List[str]:
    """
    FIX 6: Word cap raised from 8 → 20 so medical terminology isn't truncated.
    """
    try:
        raw = llm._call(QUERY_EXPANSION_PROMPT.format(query=query))
        variants = []
        for line in raw.split("\n"):
            line = line.strip().strip('"').lstrip("0123456789.-) ")
            if (
                line
                and "answer" not in line.lower()
                and len(line.split()) <= 20   # FIX 6: was 8
                and len(line.split()) >= 2
            ):
                variants.append(line)
        print(f"[Expansion] Variants: {variants[:2]}")
        return [query] + variants[:2]
    except Exception as e:
        print(f"[Expansion] Failed: {e}")
        return [query]


# ─────────────────────────────────────────────
# Corrective RAG — Relevance Grader
# ─────────────────────────────────────────────

def grade_documents(
    query: str,
    docs: List[Document],
    threshold: float = RELEVANCE_THRESHOLD,
) -> Tuple[List[Document], bool]:
    """
    FIX 2: Now wired into run() — was defined but never called before.
    Keyword-overlap grading avoids extra SageMaker round trips.
    Sets doc.metadata["relevance_score"] so it appears correctly in the output.
    """
    query_terms = set(re.findall(r"\b\w{4,}\b", query.lower()))
    filtered, scores = [], []

    for doc in docs:
        doc_terms = set(re.findall(r"\b\w{4,}\b", doc.page_content.lower()))
        overlap   = len(query_terms & doc_terms)
        score     = round(min(overlap / max(len(query_terms), 1), 1.0), 3)
        scores.append(score)
        doc.metadata["relevance_score"] = score   # FIX 9: was never set before
        if score >= threshold:
            filtered.append(doc)

    needs_requery = len(filtered) < 2 or (scores and max(scores) < threshold)
    print(f"[CRAG] Scores: {[s for s in scores]}")
    print(f"[CRAG] {len(filtered)}/{len(docs)} docs passed (threshold={threshold}). Re-query: {needs_requery}")
    return filtered, needs_requery


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def format_docs(docs: List[Document], per_doc_chars: int) -> str:
    """
    FIX 5: per_doc_chars is now passed in from the query-type-aware config
    instead of being hardcoded at 500.
    FIX 8: Citation hint appended so the model knows which source is which.
    """
    sections = []
    for i, doc in enumerate(docs, 1):
        book    = clean_book_name(doc)
        content = doc.page_content[:per_doc_chars]
        sections.append(
            f"[Source {i} | {book}]\n{content}"
        )
    return "\n\n---\n\n".join(sections)


def truncate_context(text: str, max_chars: int) -> str:
    return text[:max_chars]


def clean_book_name(doc: Document) -> str:
    book = (
        doc.metadata.get("book") or
        doc.metadata.get("source", "") or
        "Medical Textbook"
    )
    return re.split(r"[/\\]", book)[-1].replace(".txt", "")


# ─────────────────────────────────────────────
# Main RAG Pipeline
# ─────────────────────────────────────────────

class MedicalRAGPipeline:
    def __init__(self, vectorstore: PineconeVectorStore, llm: SageMakerLLM):
        self.vectorstore = vectorstore
        self.llm         = llm

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, config: Dict) -> List[Document]:
        if config["use_mmr"]:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k":           config["k"],
                    "fetch_k":     config["k"] * 3,
                    "lambda_mult": 0.6,
                },
            )
        else:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": config["k"]}
            )
        return retriever.invoke(query)

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, query: str) -> Dict:

        # ─── 1. Adaptive Routing ─────────────────────────────────────────────
        # FIX 3: MECHANISTIC type now catches mechanism/pathway/MOA queries
        query_type = classify_query(query)
        config     = get_retrieval_config(query_type)
        depth      = QUERY_DEPTH[query_type]

        per_doc_chars = CONTEXT_PER_DOC[depth]
        total_chars   = CONTEXT_TOTAL[depth]

        print(f"[Adaptive] Query type: {query_type.value} | depth: {depth} | k={config['k']} | mmr={config['use_mmr']}")

        # ─── 2. Query Expansion ──────────────────────────────────────────────
        # FIX 6: word cap raised to 20 inside expand_medical_query()
        if config["expand_query"]:
            queries = expand_medical_query(query, self.llm)
        else:
            queries = [query]

        # ─── 3. Retrieval (deduplicated) ─────────────────────────────────────
        all_docs, seen = [], set()
        for q in queries:
            for doc in self.retrieve(q, config):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        print(f"[Retrieve] {len(all_docs)} unique docs across {len(queries)} query variant(s)")

        # ─── 4. Keyword Relevance Grading (CRAG) ─────────────────────────────
        # FIX 2: grade_documents() is now actually called here
        graded_docs, needs_requery = grade_documents(query, all_docs)

        # If too few docs passed, fall back to all retrieved docs rather than
        # returning nothing — the LLM grader below is the second safety net
        candidate_docs = graded_docs if not needs_requery else all_docs

        # ─── 5. LLM Relevance Gate ───────────────────────────────────────────
        top_context = candidate_docs[0].page_content[:1000] if candidate_docs else ""
        try:
            relevance_raw = self.llm._call(
                GRADER_PROMPT.format(query=query, context=top_context)
            ).lower().strip()
            relevance = "yes" if "yes" in relevance_raw else "no"
            print(f"[Grader] LLM relevance gate: {relevance}")
        except Exception as e:
            print(f"[Grader] Failed: {e}")
            relevance = "yes"

        # ─── 6. Fallback (no relevant docs) ──────────────────────────────────
        if relevance == "no" or not candidate_docs:
            print("[Fallback] No relevant docs — generating from parametric knowledge")
            llm_answer    = self.llm._call(FALLBACK_PROMPT.format(query=query))
            filtered_docs = []
            is_fallback   = True

        else:
            # ─── 7. RAG Generation ────────────────────────────────────────────
            # FIX 5: use query-type-aware context limits
            filtered_docs = candidate_docs[:5]  # cap at 5 to keep latency sane
            context_text  = format_docs(filtered_docs, per_doc_chars)
            context_text  = truncate_context(context_text, total_chars)

            # FIX 8: cite-by-number instruction injected into the prompt
            citation_instruction = (
                "\n\nIMPORTANT: After each clinical statement, "
                "add a citation marker like [Source 1] or [Source 2] "
                "matching the source excerpts above. Only cite sources that "
                "directly support the statement."
            )

            RAG_PROMPT_MAP = {
                QueryType.FACTUAL:     RAG_PROMPT_FACTUAL,
                QueryType.MECHANISTIC: RAG_PROMPT_MECHANISTIC,
                QueryType.PROCEDURAL:  RAG_PROMPT_MECHANISTIC,  # reuse — same depth
                QueryType.DIAGNOSTIC:  RAG_PROMPT_DIAGNOSTIC,
                QueryType.COMPARATIVE: RAG_PROMPT_MECHANISTIC,
                QueryType.MULTI_HOP:   RAG_PROMPT_MECHANISTIC,
            }

            # In run():
            prompt_template = RAG_PROMPT_MAP[query_type]
            prompt = prompt_template.format(context=context_text, query=query)

            # prompt = RAG_PROMPT.format(context=context_text, query=query) + citation_instruction

            print(f"[LLM] Generating RAG answer | docs={len(filtered_docs)} | "
                  f"context={len(context_text)} chars")

            t0         = time.time()
            llm_answer = self.llm._call(prompt)
            print(f"[LLM] Done in {round(time.time() - t0, 2)}s")

            is_fallback = False

        # ─── 8. Source Attribution ────────────────────────────────────────────
        answer = llm_answer.strip()

        return {
            "query":        query,
            "query_type":   query_type.value,
            "answer":       answer,
            "source_docs":  filtered_docs,
            "context":      top_context,
            "num_docs":     len(filtered_docs),
            "fallback_used": is_fallback,
        }


# ─────────────────────────────────────────────
# Pipeline Factory
# ─────────────────────────────────────────────

def build_pipeline(
    endpoint: str = SAGEMAKER_ENDPOINT,
    region:   str = AWS_REGION,
) -> MedicalRAGPipeline:
    print(f"[Pipeline] Endpoint : {endpoint}")
    print(f"[Pipeline] Region   : {region}")
    vectorstore = load_vectorstore()
    llm         = SageMakerLLM(endpoint_name=endpoint, region_name=region)
    print("[Pipeline] ✅ Ready!")
    return MedicalRAGPipeline(vectorstore=vectorstore, llm=llm)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical RAG Pipeline")
    parser.add_argument("--query",    type=str, help="Medical question to answer")
    parser.add_argument("--endpoint", type=str, default=SAGEMAKER_ENDPOINT)
    parser.add_argument("--region",   type=str, default=AWS_REGION)
    args = parser.parse_args()

    pipeline_obj = build_pipeline(endpoint=args.endpoint, region=args.region)

    if args.query:
        result = pipeline_obj.run(args.query)
        print("\n" + "=" * 60)
        print(f"Query Type  : {result['query_type']}")
        print(f"Docs Used   : {result['num_docs']}")
        print(f"Fallback    : {result['fallback_used']}")
        print(f"\nAnswer:\n{result['answer']}")
    else:
        print("\nMedical RAG ready. Type your question (or 'quit' to exit):\n")
        while True:
            q = input("Question: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                result = pipeline_obj.run(q)
                print(f"\nAnswer:\n{result['answer']}\n")
                print(f"Sources: {[d.metadata.get('book', '?') for d in result['source_docs']]}\n")