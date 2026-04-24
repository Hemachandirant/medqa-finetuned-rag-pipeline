"""
Medical RAG Pipeline
====================
Strategy: Hybrid Adaptive + Corrective RAG
- Adaptive RAG: Routes queries to different retrieval strategies based on query complexity
- Corrective RAG: Validates retrieved documents and re-queries with broader config if coverage is low
- Vector Store: Pinecone (cloud-hosted, accessible from any deployment)
- Embeddings: Azure OpenAI text-embedding-3-small (matches index_documents.py)
- LLM: Fine-tuned DeepSeek on AWS SageMaker (ap-south-1)

RETRIEVAL ACCURACY IMPROVEMENTS
================================
R01  CrossEncoderReranker replaces keyword-overlap grading as the primary relevance filter.
     Model preference order: BAAI/bge-reranker-base (primary — better on biomedical text),
     cross-encoder/ms-marco-MiniLM-L-6-v2 (fallback — faster, web-domain trained).
     Scores are sigmoid-normalised to [0, 1].

ORIGINAL FIXES (unchanged)
===========================
F01  PrivateAttr used for _runtime — prevents Pydantic from trying to validate the boto3 client
F02  model_post_init calls super() before the boto3 initialisation
F03  boto3 credential chain used — no hardcoded key/secret; IAM roles work automatically
F04  Logging replaces all print() calls — level-controllable at runtime
F05  Deduplication uses MD5 of full page_content instead of first 100 chars
F06  grade_documents uses \\w{2,} regex to catch short medical abbreviations (MI, DVT, CBC…)
F07  grade_documents returns new Document objects — original metadata is never mutated
F08  Actual corrective re-query: when coverage is low, a second pass with doubled k and
     similarity search (no MMR) is run before falling back to all_docs
F09  Multi-query retrieval is threaded with ThreadPoolExecutor — ~3× latency reduction
F10  candidate_docs cap uses config["k"] instead of the hardcoded magic number 5
F11  citation_instruction dead variable removed — citation text is now embedded in every prompt
F12  RAG_PROMPT_MAP updated: PROCEDURAL → RAG_PROMPT_PROCEDURAL,
                              COMPARATIVE → RAG_PROMPT_COMPARATIVE,
                              MULTI_HOP stays MECHANISTIC (deep, multi-step reasoning)
F13  _generate passes run_manager through to _call for LangChain callback/tracing support
F14  QUERY_EXPANSION_PROMPT now says "exactly 2" to match the [:2] slice
F15  Context limits expressed in approximate tokens (chars // 4) via _approx_tokens(),
     making it easy to swap in a real tokenizer later without touching call-sites
"""

import os
import re
import json
import time
import hashlib
import logging
import boto3
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import Field, PrivateAttr  # F01

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from prompts.medical_prompts import (
    SYSTEM_PROMPT,
    RAG_PROMPT_FACTUAL,
    RAG_PROMPT_MECHANISTIC,
    RAG_PROMPT_PROCEDURAL,
    RAG_PROMPT_DIAGNOSTIC,
    RAG_PROMPT_COMPARATIVE,
    QUERY_EXPANSION_PROMPT,
    GRADER_PROMPT,
    FALLBACK_PROMPT,
)


# ─────────────────────────────────────────────
# F04 — module-level logger
# ─────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration (loaded from .env)
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

# ── R01/R02: Cross-encoder + pre-filter thresholds ──────────────────────────
# RERANK_THRESHOLD: sigmoid(score) below this → doc is dropped entirely.
# 0.30 is a safe starting point for medical Q&A.  Raise to 0.40 if you see
# too many borderline docs; lower to 0.20 if recall matters more than precision.
RERANK_THRESHOLD = 0.30

# GENERATION_RELEVANCE_THRESHOLD: secondary filter applied *after* reranking,
# immediately before the generation context is assembled.
#
# Motivation: a doc at 0.31 clears RERANK_THRESHOLD (used for CRAG decisions)
# but can still pollute the generation prompt when it contains side-by-side
# normal/abnormal tables — the LLM sees both columns and conflates them.
#
# Docs below this score are excluded from the context passed to the LLM.
# The top-1 doc is always kept regardless, so this never produces zero context.
# Rule of thumb: set ~0.15–0.20 above RERANK_THRESHOLD.
GENERATION_RELEVANCE_THRESHOLD = 0.50

# KEYWORD_PREFILTER_THRESHOLD: cheap keyword overlap floor before cross-encoder.
# Docs scoring below this skip the (slower) neural scorer entirely.
# Set to 0 to disable pre-filtering and always run full cross-encoder.
KEYWORD_PREFILTER_THRESHOLD = 0.05

# Cross-encoder model names in preference order.
# ms-marco is faster on CPU; bge-reranker-base scores slightly higher on biomedical text.
CROSS_ENCODER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # primary  — ~22 ms/pair CPU
    "BAAI/bge-reranker-base",                  # fallback — heavier but better on bio text
]

# F15 — context limits expressed as approximate token counts.
CONTEXT_PER_DOC_TOKENS = {
    "shallow": 150,   # ≈600 chars
    "medium":  225,   # ≈900 chars
    "deep":    300,   # ≈1200 chars
}
CONTEXT_TOTAL_TOKENS = {
    "shallow": 1000,  # ≈4000 chars
    "medium":  1500,  # ≈6000 chars
    "deep":    2000,  # ≈8000 chars
}

CHARS_PER_TOKEN = 4  # rough English medical prose average


def _approx_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters for English medical text."""
    return len(text) // CHARS_PER_TOKEN


def _tokens_to_chars(tokens: int) -> int:
    return tokens * CHARS_PER_TOKEN


# ─────────────────────────────────────────────
# R01 — Cross-Encoder Reranker
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for relevance scoring.

    Usage
    -----
    reranker = CrossEncoderReranker()              # loads model once at startup
    scored   = reranker.rerank(query, docs)        # [(doc, float), …] sorted best-first
    filtered = reranker.filter(query, docs)        # docs above RERANK_THRESHOLD only

    Design notes
    ------------
    • Model is loaded lazily on first call so import stays fast.
    • sigmoid() is applied to raw logits so every score is in [0, 1] and
      comparable across query types.
    • Scores are attached to doc.metadata["rerank_score"] in a NEW Document
      object (F07-style: originals never mutated).
    • If the model cannot be loaded (network issue, missing package) the reranker
      degrades gracefully to the keyword pre-filter score already attached.
    """

    def __init__(
        self,
        model_names: List[str] = CROSS_ENCODER_MODELS,
        threshold: float = RERANK_THRESHOLD,
    ):
        self._model_names = model_names
        self.threshold = threshold
        self._model = None          # lazy-loaded
        self._loaded_model_name: Optional[str] = None

    # ── lazy model load ───────────────────────────────────────────────────────

    def _load_model(self):
        """Try each model in preference order; warn and set to None if all fail."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers\n"
                "Reranker will be disabled — keyword pre-filter scores used instead."
            )
            return
        for name in self._model_names:
            try:
                self._model = CrossEncoder(name, max_length=512)
                self._loaded_model_name = name
                logger.info("CrossEncoder loaded: %s", name)
                return
            except Exception as e:
                logger.warning("Could not load cross-encoder '%s': %s", name, e)
        logger.error("All cross-encoder models failed to load — reranker disabled.")

    # ── scoring ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Normalise raw logit to [0, 1]."""
        return float(1.0 / (1.0 + np.exp(-x)))

    def score(self, query: str, docs: List[Document]) -> List[float]:
        """
        Return a sigmoid-normalised relevance score for each doc.
        Falls back to metadata["keyword_score"] (set by pre-filter) if model unavailable.
        """
        self._load_model()
        if self._model is None:
            # graceful degradation: use keyword score already stored by pre-filter
            return [
                float(d.metadata.get("keyword_score", 0.0)) for d in docs
            ]
        pairs = [(query, doc.page_content[:512]) for doc in docs]
        try:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
            return [self._sigmoid(float(s)) for s in raw_scores]
        except Exception as e:
            logger.warning("CrossEncoder.predict failed: %s — falling back to keyword scores", e)
            return [float(d.metadata.get("keyword_score", 0.0)) for d in docs]

    # ── public API ─────────────────────────────────────────────────────────────

    def rerank(
        self, query: str, docs: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Score all docs and return [(doc, score), …] sorted best-first.

        Metadata written to every returned Document (new object, F07 — originals never mutated):
          relevance_score  – sigmoid-normalised float in [0, 1].  This is the canonical
                             key read by the serving/serialisation layer.
          score_source     – "cross_encoder" | "keyword_fallback" so callers know which
                             scoring path fired (useful when debugging zero scores).
        """
        if not docs:
            return []
        scores      = self.score(query, docs)
        score_src   = (
            "cross_encoder"   if self._model is not None else "keyword_fallback"
        )
        pairs = []
        for doc, score in zip(docs, scores):
            scored_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "relevance_score": round(score, 4),   # ← canonical key
                    "score_source":    score_src,
                },
            )
            pairs.append((scored_doc, score))
        pairs.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "Reranker scores (%s): %s",
            score_src,
            [round(p[1], 4) for p in pairs],
        )
        return pairs

    def filter(
        self, query: str, docs: List[Document]
    ) -> List[Document]:
        """
        R01: Return only docs whose rerank score >= threshold.
        If NOTHING clears the threshold, return an empty list — the pipeline
        will then trigger its fallback rather than answer from irrelevant context.
        """
        ranked = self.rerank(query, docs)
        accepted = [doc for doc, score in ranked if score >= self.threshold]
        rejected = len(ranked) - len(accepted)
        logger.info(
            "CrossEncoder filter | model=%s threshold=%.2f | %d/%d docs accepted, %d rejected",
            self._loaded_model_name or "degraded",
            self.threshold,
            len(accepted),
            len(docs),
            rejected,
        )
        return accepted


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
    logger.info("Connecting to Pinecone index '%s'…", PINECONE_INDEX_NAME)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
    )
    logger.info("Pinecone connected.")
    return vectorstore


# ─────────────────────────────────────────────
# SageMaker LLM Wrapper (unchanged from original)
# ─────────────────────────────────────────────

class SageMakerLLM(BaseLLM):
    """LangChain-compatible LLM wrapper for fine-tuned DeepSeek on SageMaker."""

    endpoint_name:  str = Field(default=SAGEMAKER_ENDPOINT)
    region_name:    str = Field(default=AWS_REGION)
    max_new_tokens: int = Field(default=1024)
    max_retries:    int = Field(default=3)

    _runtime: object = PrivateAttr(default=None)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._runtime = boto3.client(
            "sagemaker-runtime",
            region_name=self.region_name,
        )

    @property
    def _llm_type(self) -> str:
        return "sagemaker-deepseek"

    def _build_prompt(self, user_message: str, prefix: str = "") -> str:
        base = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return base if not prefix else base + prefix

    def _invoke_with_retry(self, payload: dict) -> str:
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
                logger.warning("SageMaker throttled (attempt %d/%d). Retrying in %ds…",
                               attempt + 1, self.max_retries, wait)
                time.sleep(wait)
                last_error = e
            except Exception as e:
                logger.error("SageMaker non-retryable error: %s", e)
                raise
        raise RuntimeError(
            f"SageMaker: all {self.max_retries} retries exhausted. Last error: {last_error}"
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"(?i)^the answer (to (your|this) (question|query) )?is[:\s]*", "", text)
        return text.strip()

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
            logger.error("SageMaker _call failed: %s", e)
            return "Error: Could not get a response from the model endpoint."

    def _call_with_prefix(self, user_message: str, prefix: str) -> str:
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
            logger.error("SageMaker _call_with_prefix failed: %s", e)
            return "Error: Could not retrieve a response from the model endpoint."

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import LLMResult, Generation
        generations = []
        for i, p in enumerate(prompts):
            child_manager = run_manager.get_child() if run_manager else None
            text = self._call(p, stop=stop, run_manager=child_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


# ─────────────────────────────────────────────
# Query Complexity Classifier (Adaptive RAG)
# ─────────────────────────────────────────────

class QueryType(Enum):
    FACTUAL     = "factual"
    PROCEDURAL  = "procedural"
    MECHANISTIC = "mechanistic"
    DIAGNOSTIC  = "diagnostic"
    COMPARATIVE = "comparative"
    MULTI_HOP   = "multi_hop"


QUERY_PATTERNS: Dict[QueryType, List[str]] = {
    QueryType.MECHANISTIC: [
        r"\bmechanism\b", r"\bmoa\b", r"\bmode of action\b",
        r"\bpathway\b",   r"\bhow does .+ work\b",
        r"\bpharmacodynamic\b", r"\bpharmacology\b",
        r"\bactivat(e|es|ion)\b", r"\binhibit(s|ion)\b",
        r"\bagonist\b",   r"\bantagonist\b",
        r"\breceptor\b",  r"\bsignalling\b",
    ],
    # DIAGNOSTIC is checked before FACTUAL so "what are the criteria for diagnosis"
    # is caught here rather than falling through to the "what" pattern below.
    QueryType.DIAGNOSTIC: [
        r"\bdifferential\b",
        r"\bdiagnose\b", r"\bdiagnosis\b", r"\bdiagnostic\b",
        r"\bcriteria\b", r"\bclassif(y|ication)\b",
        r"\bpresent(s|ing)?\b", r"\bsymptom\b",
        r"\bsign(s)?\b",
        r"\bscreening\b", r"\bworkup\b",
    ],
    QueryType.FACTUAL: [
        r"\bwhat (is|are)\b", r"\bdefine\b", r"\bnormal range\b", r"\bvalue of\b",
        r"\blist (the|of)\b",
    ],
    QueryType.PROCEDURAL: [
        r"\bhow (is|to|do)\b", r"\bprocedure\b", r"\bsteps\b", r"\bperform\b",
        r"\bmanag(e|ement)\b", r"\btreat(ment|ing)?\b",
    ],
    QueryType.COMPARATIVE: [
        r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bdifference\b",
    ],
    QueryType.MULTI_HOP: [
        r"\baffect\b", r"\brelationship\b", r"\bimpact\b", r"\binteraction\b",
    ],
}

QUERY_DEPTH: Dict[QueryType, str] = {
    # "shallow" (600 chars/doc) truncates criteria tables and value ranges mid-sentence.
    # FACTUAL raised to "medium" (900 chars) — still fast, prevents mid-sentence cuts.
    QueryType.FACTUAL:     "medium",
    QueryType.PROCEDURAL:  "medium",
    QueryType.MECHANISTIC: "deep",
    # Diagnostic criteria are dense structured tables — needs full "deep" context.
    QueryType.DIAGNOSTIC:  "deep",
    QueryType.COMPARATIVE: "deep",
    QueryType.MULTI_HOP:   "deep",
}


def classify_query(query: str) -> QueryType:
    q = query.lower()
    for qtype, patterns in QUERY_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            return qtype
    return QueryType.FACTUAL


def get_retrieval_config(query_type: QueryType) -> Dict:
    return {
        QueryType.FACTUAL:     {"k": 5,  "use_mmr": False, "expand_query": False},
        QueryType.PROCEDURAL:  {"k": 5,  "use_mmr": True,  "expand_query": False},
        QueryType.MECHANISTIC: {"k": 8,  "use_mmr": True,  "expand_query": True},
        QueryType.DIAGNOSTIC:  {"k": 7,  "use_mmr": True,  "expand_query": True},
        QueryType.COMPARATIVE: {"k": 8,  "use_mmr": True,  "expand_query": True},
        QueryType.MULTI_HOP:   {"k": 10, "use_mmr": True,  "expand_query": True},
    }[query_type]


# ─────────────────────────────────────────────
# Query Expansion
# ─────────────────────────────────────────────

def expand_medical_query(query: str, llm: SageMakerLLM) -> List[str]:
    try:
        raw = llm._call(QUERY_EXPANSION_PROMPT.format(query=query))
        variants = []
        for line in raw.split("\n"):
            line = line.strip().strip('"').lstrip("0123456789.-) ")
            if (
                line
                and "answer" not in line.lower()
                and 2 <= len(line.split()) <= 20
            ):
                variants.append(line)
        logger.debug("Expanded query variants: %s", variants[:2])
        return [query] + variants[:2]
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
        return [query]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _doc_key(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def clean_book_name(doc: Document) -> str:
    book = (
        doc.metadata.get("book") or
        doc.metadata.get("source", "") or
        "Medical Textbook"
    )
    return re.split(r"[/\\]", book)[-1].replace(".txt", "")


def format_docs(docs: List[Document], per_doc_tokens: int) -> str:
    per_doc_chars = _tokens_to_chars(per_doc_tokens)
    sections = []
    for i, doc in enumerate(docs, 1):
        book    = clean_book_name(doc)
        content = doc.page_content[:per_doc_chars]
        # relevance_score is the canonical metadata key set by CrossEncoderReranker.rerank()
        score   = doc.metadata.get("relevance_score", "?")
        src     = doc.metadata.get("score_source", "")
        sections.append(f"[Source {i} | {book} | relevance={score} ({src})]\n{content}")
    return "\n\n---\n\n".join(sections)


def truncate_context(text: str, max_tokens: int) -> str:
    """
    Truncate context to approximate token budget while respecting sentence boundaries.

    Hard character-slicing (the previous implementation) could cut a criteria table
    mid-value, e.g. "≥7.0 mmol/" — leaving the LLM to hallucinate the missing unit.
    This version walks back from the hard limit to the nearest sentence-ending
    punctuation so the last sentence in context is always complete.
    """
    max_chars = _tokens_to_chars(max_tokens)
    if len(text) <= max_chars:
        return text
    # Walk back from the limit to find the last sentence boundary
    truncated = text[:max_chars]
    # Look for ". ", ".\n", "? ", "! " within the last 20 % of the window
    search_from = int(max_chars * 0.80)
    last_boundary = max(
        truncated.rfind(". ",  search_from),
        truncated.rfind(".\n", search_from),
        truncated.rfind("? ",  search_from),
        truncated.rfind("! ",  search_from),
    )
    if last_boundary != -1:
        return truncated[: last_boundary + 1]   # include the full stop
    # No sentence boundary found in the search window — fall back to hard cut
    return truncated


# ─────────────────────────────────────────────
# R02 — Keyword Pre-filter (Stage 1 of 2-stage funnel)
# ─────────────────────────────────────────────

_STOPWORDS = frozenset({
    "is", "in", "it", "of", "to", "a", "an", "the", "and", "or",
    "for", "on", "at", "be", "by", "as", "do", "if", "no", "so",
    "what", "how", "which", "when", "where", "why", "does", "did",
    "are", "was", "were", "has", "have", "had", "can", "could",
    "should", "would", "will", "may", "might", "its", "their",
    "this", "that", "with", "from", "about", "than", "not",
})


def _keyword_prefilter(
    query: str,
    docs: List[Document],
    threshold: float = KEYWORD_PREFILTER_THRESHOLD,
) -> List[Document]:
    """
    R02 Stage 1: cheap keyword overlap to eliminate obviously irrelevant docs
    before the cross-encoder runs.  Stores keyword_score in metadata so the
    reranker can use it as a fallback if the model fails to load.

    This step is intentionally lenient (threshold=0.05 by default) — it only
    removes docs with near-zero topical overlap.
    """
    query_terms = {
        t for t in re.findall(r"\b\w{2,}\b", query.lower())
        if t not in _STOPWORDS
    }
    if not query_terms:
        return docs  # can't filter without query terms

    kept = []
    for doc in docs:
        doc_terms = {
            t for t in re.findall(r"\b\w{2,}\b", doc.page_content.lower())
            if t not in _STOPWORDS
        }
        overlap = len(query_terms & doc_terms)
        score   = round(min(overlap / max(len(query_terms), 1), 1.0), 3)
        scored_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "keyword_score": score},
        )
        if score >= threshold:
            kept.append(scored_doc)

    removed = len(docs) - len(kept)
    logger.debug(
        "Keyword pre-filter: %d/%d docs kept (threshold=%.2f, removed=%d)",
        len(kept), len(docs), threshold, removed,
    )
    return kept


# ─────────────────────────────────────────────
# R03 — Unified grade_documents (Stage 1 + 2)
# ─────────────────────────────────────────────

def grade_documents(
    query: str,
    docs: List[Document],
    reranker: "CrossEncoderReranker",
    threshold: float = RELEVANCE_THRESHOLD,
) -> Tuple[List[Document], bool]:
    """
    Two-stage relevance filtering (R02/R03).

    Stage 1 – keyword pre-filter (fast, removes obvious noise)
    Stage 2 – cross-encoder reranker (semantic, hard threshold)

    Returns
    -------
    (filtered_docs, needs_requery)
        filtered_docs  : semantically relevant docs, sorted best-first.
                         Empty list if nothing is relevant — pipeline falls back.
        needs_requery  : True when fewer than 2 docs cleared the bar
                         (signals CRAG corrective re-query).
    """
    if not docs:
        return [], True

    # Stage 1 — keyword pre-filter
    pre_filtered = _keyword_prefilter(query, docs)
    logger.debug("After keyword pre-filter: %d docs remain", len(pre_filtered))

    # Stage 2 — cross-encoder
    # Use the reranker's own threshold (RERANK_THRESHOLD), not RELEVANCE_THRESHOLD,
    # so the two constants can be tuned independently.
    filtered = reranker.filter(query, pre_filtered)

    needs_requery = len(filtered) < 2
    logger.info(
        "grade_documents | %d/%d passed cross-encoder | needs_requery=%s",
        len(filtered), len(docs), needs_requery,
    )
    return filtered, needs_requery


# ─────────────────────────────────────────────
# Main RAG Pipeline
# ─────────────────────────────────────────────

RAG_PROMPT_MAP = {
    QueryType.FACTUAL:     RAG_PROMPT_FACTUAL,
    QueryType.MECHANISTIC: RAG_PROMPT_MECHANISTIC,
    QueryType.PROCEDURAL:  RAG_PROMPT_PROCEDURAL,
    QueryType.DIAGNOSTIC:  RAG_PROMPT_DIAGNOSTIC,
    QueryType.COMPARATIVE: RAG_PROMPT_COMPARATIVE,
    QueryType.MULTI_HOP:   RAG_PROMPT_MECHANISTIC,
}


class MedicalRAGPipeline:
    def __init__(
        self,
        vectorstore: PineconeVectorStore,
        llm: SageMakerLLM,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.vectorstore = vectorstore
        self.llm         = llm
        # R01: reranker is created here if not injected (allows mock in tests)
        self.reranker    = reranker or CrossEncoderReranker()

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

    # ── threaded multi-query retrieval ────────────────────────────────────────

    def _retrieve_all(
        self,
        queries: List[str],
        config: Dict,
        seen: set,
    ) -> List[Document]:
        """F09: concurrent retrieval across query variants."""
        results: List[Document] = []
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
            futures = {pool.submit(self.retrieve, q, config): q for q in queries}
            for future in as_completed(futures):
                try:
                    for doc in future.result():
                        key = _doc_key(doc)
                        if key not in seen:
                            seen.add(key)
                            results.append(doc)
                except Exception as e:
                    logger.warning("Retrieval future raised: %s", e)
        return results

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, query: str) -> Dict:

        # ─── 1. Adaptive Routing ─────────────────────────────────────────────
        query_type = classify_query(query)
        config     = get_retrieval_config(query_type)
        depth      = QUERY_DEPTH[query_type]

        per_doc_tokens = CONTEXT_PER_DOC_TOKENS[depth]
        total_tokens   = CONTEXT_TOTAL_TOKENS[depth]

        logger.info(
            "Adaptive routing | type=%s depth=%s k=%d mmr=%s",
            query_type.value, depth, config["k"], config["use_mmr"],
        )

        # ─── 2. Query Expansion ──────────────────────────────────────────────
        queries = (
            expand_medical_query(query, self.llm)
            if config["expand_query"]
            else [query]
        )

        # ─── 3. Threaded Retrieval (deduplicated) ────────────────────────────
        seen: set = set()
        all_docs  = self._retrieve_all(queries, config, seen)
        logger.info(
            "Retrieved %d unique docs across %d query variant(s)",
            len(all_docs), len(queries),
        )

        # ─── 4. Two-stage Relevance Grading (keyword pre-filter → cross-encoder)
        graded_docs, needs_requery = grade_documents(
            query, all_docs, self.reranker
        )

        # ─── 5. Corrective Re-query ───────────────────────────────────────────
        # F08 + R03: trigger CRAG when fewer than 2 docs cleared cross-encoder.
        # Re-query with doubled k and plain similarity search, then re-grade.
        if needs_requery:
            logger.info("CRAG: low coverage — running corrective re-query…")
            requery_config = {
                **config,
                "k":       min(config["k"] * 2, TOP_K_EXPANDED),
                "use_mmr": False,
            }
            extra_docs = self._retrieve_all(queries, requery_config, seen)
            all_docs.extend(extra_docs)
            logger.info("Re-query added %d new docs (total pool: %d)", len(extra_docs), len(all_docs))

            # Re-grade the extended pool through the full two-stage funnel
            graded_docs, _ = grade_documents(query, all_docs, self.reranker)

        # ─── 6. Hard stop: nothing relevant found ────────────────────────────
        # R03: if cross-encoder still finds nothing relevant after re-query,
        # do NOT pass noisy context to the LLM — go straight to fallback.
        if not graded_docs:
            logger.info(
                "No documents cleared the relevance threshold after re-query. "
                "Falling back to parametric generation."
            )
            llm_answer  = self.llm._call(FALLBACK_PROMPT.format(query=query))
            return {
                "query":         query,
                "query_type":    query_type.value,
                "answer":        llm_answer.strip(),
                "source_docs":   [],
                "context":       "",
                "num_docs":      0,
                "fallback_used": True,
            }

        candidate_docs = graded_docs  # already sorted best-first by reranker

        # ─── 7. LLM Relevance Gate (lightweight sanity check) ────────────────
        # Now only checks top-3 snippets; cross-encoder already did the heavy lifting.
        top_context = "\n\n".join(
            d.page_content[:300] for d in candidate_docs[:3]
        )
        try:
            relevance_raw = self.llm._call(
                GRADER_PROMPT.format(query=query, context=top_context)
            ).lower().strip()
            relevance = "yes" if "yes" in relevance_raw else "no"
            logger.info("LLM relevance gate: %s", relevance)
        except Exception as e:
            logger.warning("LLM grader failed (%s) — defaulting to 'yes'", e)
            relevance = "yes"

        # ─── 8. Fallback (LLM gate says no) ──────────────────────────────────
        if relevance == "no":
            logger.info("LLM gate rejected docs — falling back to parametric generation")
            llm_answer    = self.llm._call(FALLBACK_PROMPT.format(query=query))
            filtered_docs = []
            is_fallback   = True

        else:
            # ─── 9. RAG Generation ────────────────────────────────────────────

            # Apply generation threshold: remove borderline docs that cleared
            # RERANK_THRESHOLD (kept for CRAG decisions) but are too noisy to
            # include in the prompt.  Always keep at least the top-1 doc.
            gen_docs = [
                d for d in candidate_docs
                if d.metadata.get("relevance_score", 0.0) >= GENERATION_RELEVANCE_THRESHOLD
            ]
            if not gen_docs:
                # Everything is borderline — keep only the single best doc
                gen_docs = candidate_docs[:1]
                logger.info(
                    "All docs below GENERATION_RELEVANCE_THRESHOLD (%.2f) — "
                    "using top-1 doc only (score=%.4f)",
                    GENERATION_RELEVANCE_THRESHOLD,
                    candidate_docs[0].metadata.get("relevance_score", 0.0),
                )

            filtered_docs = gen_docs[:config["k"]]

            logger.info(
                "Generation context: %d/%d docs after generation threshold (%.2f)",
                len(filtered_docs), len(candidate_docs), GENERATION_RELEVANCE_THRESHOLD,
            )

            # Numeric precision directive prepended to context so the model
            # sees it immediately before the source text.
            # Injected here rather than in the prompt template so it applies
            # to all query types without modifying every template.
            NUMERIC_PRECISION_DIRECTIVE = (
                "IMPORTANT — NUMERIC PRECISION RULE:\n"
                "All threshold operators (≥, >, ≤, <, =) and numeric values "
                "in the sources below MUST be reproduced exactly as written. "
                "Do NOT paraphrase, invert, or round any threshold. "
                "If a criterion says '≥ 6.5%' write '≥ 6.5%', not '> 6.5%' or '< 6.5%'.\n\n"
            )

            context_text  = format_docs(filtered_docs, per_doc_tokens)
            context_text  = truncate_context(context_text, total_tokens)
            context_text  = NUMERIC_PRECISION_DIRECTIVE + context_text

            prompt_template = RAG_PROMPT_MAP[query_type]
            prompt          = prompt_template.format(context=context_text, query=query)

            logger.info(
                "Generating RAG answer | docs=%d context≈%d tokens",
                len(filtered_docs), _approx_tokens(context_text),
            )

            t0         = time.time()
            llm_answer = self.llm._call(prompt)
            logger.info("Generation complete in %.2fs", time.time() - t0)

            is_fallback = False

        # ─── 10. Return ───────────────────────────────────────────────────────
        return {
            "query":         query,
            "query_type":    query_type.value,
            "answer":        llm_answer.strip(),
            "source_docs":   filtered_docs,
            "context":       top_context,
            "num_docs":      len(filtered_docs),
            "fallback_used": is_fallback,
        }


# ─────────────────────────────────────────────
# Pipeline Factory
# ─────────────────────────────────────────────

def build_pipeline(
    endpoint: str = SAGEMAKER_ENDPOINT,
    region:   str = AWS_REGION,
) -> MedicalRAGPipeline:
    logger.info("Building pipeline | endpoint=%s region=%s", endpoint, region)
    vectorstore = load_vectorstore()
    llm         = SageMakerLLM(endpoint_name=endpoint, region_name=region)
    reranker    = CrossEncoderReranker()   # R01: instantiate once, reuse across requests
    logger.info("Pipeline ready.")
    return MedicalRAGPipeline(vectorstore=vectorstore, llm=llm, reranker=reranker)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

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