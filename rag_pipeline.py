"""
Medical RAG Pipeline
====================
Strategy: Hybrid Adaptive + Corrective RAG
- Adaptive RAG: Routes queries to different retrieval strategies based on query complexity
- Corrective RAG: Validates retrieved documents and re-queries if relevance is low
- Vector Store: Pinecone (cloud-hosted, accessible from any deployment)
- Embeddings: Azure OpenAI text-embedding-3-small (matches index_documents.py)
- LLM: Fine-tuned DeepSeek on AWS SageMaker (ap-south-1)
"""

import os
import re
import json
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
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

from prompts.medical_prompts import (
    SYSTEM_PROMPT,
    RAG_PROMPT,
    QUERY_EXPANSION_PROMPT,
    GRADER_PROMPT,
    FALLBACK_PROMPT
)
# ─────────────────────────────────────────────
# Configuration (load from .env)
# ─────────────────────────────────────────────

SAGEMAKER_ENDPOINT  = os.getenv("SAGEMAKER_ENDPOINT_NAME", "medqa-deepseek-v2")
AWS_REGION          = os.getenv("AWS_REGION", "ap-south-1")

# Pinecone
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medqa-textbooks")

# Azure OpenAI Embeddings — must match what was used in index_documents.py
AZURE_ENDPOINT      = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY       = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION   = os.getenv("AZURE_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT    = os.getenv("AZURE_DEPLOYMENT", "text-embedding-3-small")

TOP_K_INITIAL       = 5
TOP_K_EXPANDED      = 10
RELEVANCE_THRESHOLD = 0.35

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Answer questions in detail with drug names, mechanisms, and clinical reasoning."
)


# ─────────────────────────────────────────────
# Embeddings — must match index_documents.py
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
    embeddings  = get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    print(f"[Pinecone] ✅ Connected.")
    return vectorstore


# ─────────────────────────────────────────────
# SageMaker LLM Wrapper
# ─────────────────────────────────────────────

class SageMakerLLM(BaseLLM):
    """
    LangChain-compatible LLM wrapper for fine-tuned DeepSeek
    deployed at: https://runtime.sagemaker.ap-south-1.amazonaws.com/
                 endpoints/medqa-deepseek-v2/invocations
    """
    endpoint_name:  str = Field(default=SAGEMAKER_ENDPOINT)
    region_name:    str = Field(default=AWS_REGION)
    max_new_tokens: int = Field(default=1024)

    @property
    def _llm_type(self) -> str:
        return "sagemaker-deepseek"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        # DeepSeek-R1-Distill-Qwen — Test 2 format (proven to give detailed answers)
        formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        payload = {
            "inputs": formatted,
            "parameters": {
                "max_new_tokens":     self.max_new_tokens,
                "temperature":        0.2,   # Low temperature for factual medical QA
                "top_p":              0.95,
                "repetition_penalty": 1.05,
                "do_sample":          True,
                "return_full_text":   False,
            }
        }

        try:
            response = runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            result = json.loads(response["Body"].read().decode())
            text   = result[0]["generated_text"] if isinstance(result, list) else str(result)

            # DeepSeek-R1 emits <think>...</think> reasoning tokens before the answer
            # Strip them so only the final answer is returned
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Also strip any leftover im_end tokens
            text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL).strip()

            return text

        except Exception as e:
            print(f"[SageMaker] Error calling endpoint: {e}")
            return "Error: Could not get response from model endpoint."

    def _call_with_prefix(self, user_message: str, prefix: str) -> str:
        """
        Injects an answer prefix directly into the assistant turn.
        Forces the model to continue the prefix instead of starting fresh.
        """
        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n{prefix}"  # model continues from here
        )

        payload = {
            "inputs": formatted,
            "parameters": {
                "max_new_tokens":     self.max_new_tokens,
                "temperature":        0.2,   # Low temperature for factual medical QA
                "top_p":              0.95,
                "repetition_penalty": 1.05,
                "do_sample":          True,
                "return_full_text":   False,
            }
        }

        try:
            response = runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            result = json.loads(response["Body"].read().decode())
            text   = result[0]["generated_text"] if isinstance(result, list) else str(result)
            text   = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text   = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as e:
            print(f"[SageMaker] Error: {e}")
            return "could not retrieve a response from the model endpoint."

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
    DIAGNOSTIC  = "diagnostic"
    COMPARATIVE = "comparative"
    MULTI_HOP   = "multi_hop"


QUERY_PATTERNS = {
    QueryType.FACTUAL:     [r"\bwhat is\b", r"\bdefine\b", r"\bnormal range\b", r"\bvalue of\b"],
    QueryType.PROCEDURAL:  [r"\bhow (is|to|do)\b", r"\bprocedure\b", r"\bsteps\b", r"\bperform\b"],
    QueryType.DIAGNOSTIC:  [r"\bdifferential\b", r"\bdiagnose\b", r"\bpresent(s|ing)?\b", r"\bsymptom\b"],
    QueryType.COMPARATIVE: [r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bdifference\b"],
    QueryType.MULTI_HOP:   [r"\baffect\b", r"\brelationship\b", r"\bimpact\b", r"\binteraction\b"],
}


def classify_query(query: str) -> QueryType:
    q = query.lower()
    for qtype, patterns in QUERY_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            return qtype
    return QueryType.FACTUAL


def get_retrieval_config(query_type: QueryType) -> Dict:
    configs = {
        QueryType.FACTUAL:     {"k": 5,  "use_mmr": False, "expand_query": False},
        QueryType.PROCEDURAL:  {"k": 5,  "use_mmr": True,  "expand_query": False},
        QueryType.DIAGNOSTIC:  {"k": 7,  "use_mmr": True,  "expand_query": True},
        QueryType.COMPARATIVE: {"k": 8,  "use_mmr": True,  "expand_query": True},
        QueryType.MULTI_HOP:   {"k": 10, "use_mmr": True,  "expand_query": True},
    }
    return configs[query_type]


# ─────────────────────────────────────────────
# Query Expansion
# ─────────────────────────────────────────────

def expand_medical_query(query: str, llm: SageMakerLLM) -> List[str]:
    message = (
        f"Generate 3 alternative medical search queries for better document retrieval.\n"
        f"Return only the queries, one per line, no numbering.\n\n"
        f"Original query: {query}\n"
        f"Alternative queries:"
    )
    result   = llm._call(message)
    variants = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return [query] + variants[:3]


# ─────────────────────────────────────────────
# Corrective RAG — Relevance Grader
# ─────────────────────────────────────────────

def grade_documents(
    query: str,
    docs: List[Document],
    llm: SageMakerLLM,
    threshold: float = RELEVANCE_THRESHOLD,
) -> Tuple[List[Document], bool]:
    """
    Fast relevance grading using keyword overlap — no LLM calls.
    Avoids N extra SageMaker round trips before answering.
    """
    query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
    filtered, scores = [], []

    for doc in docs:
        doc_terms  = set(re.findall(r'\b\w{4,}\b', doc.page_content.lower()))
        overlap    = len(query_terms & doc_terms)
        score      = min(overlap / max(len(query_terms), 1), 1.0)
        scores.append(score)
        doc.metadata["relevance_score"] = round(score, 3)
        if score >= threshold:
            filtered.append(doc)

    needs_requery = len(filtered) < 2 or (scores and max(scores) < threshold)
    print(f"[CRAG] Scores: {[round(s,2) for s in scores]}")
    print(f"[CRAG] {len(filtered)}/{len(docs)} docs passed (threshold={threshold}). Re-query: {needs_requery}")
    return filtered, needs_requery


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    sections = []
    for i, doc in enumerate(docs, 1):
        book = doc.metadata.get(
            "book",
            Path(doc.metadata.get("source", "Unknown")).stem
        )

        # 🔥 LIMIT EACH DOC SIZE
        content = doc.page_content[:500]

        sections.append(f"[Source {i} | {book}]\n{content}")

    return "\n\n---\n\n".join(sections)

def truncate_context(text: str, max_chars: int = 3500) -> str:
    return text[:max_chars]


def clean_book_name(doc: Document) -> str:
    book = (
        doc.metadata.get("book") or
        doc.metadata.get("source", "") or
        "Medical Textbook"
    )
    book = re.split(r"[/\\]", book)[-1].replace(".txt", "")
    return book


# ─────────────────────────────────────────────
# Main RAG Pipeline
# ─────────────────────────────────────────────

class MedicalRAGPipeline:
    def __init__(self, vectorstore: PineconeVectorStore, llm: SageMakerLLM):
        self.vectorstore = vectorstore
        self.llm         = llm

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

    def run(self, query: str) -> Dict:
        # ─────────────────────────────
        # 1. Adaptive Routing
        # ─────────────────────────────
        query_type = classify_query(query)
        config = get_retrieval_config(query_type)

        print(f"[Adaptive] Query type: {query_type.value} | k={config['k']} | mmr={config['use_mmr']}")

        # ─────────────────────────────
        # 2. Query Expansion (NEW PROMPT)
        # ─────────────────────────────
        try:
            expansion_msg = QUERY_EXPANSION_PROMPT.format(query=query)
            raw = self.llm._call(expansion_msg)

            variants = []
            for line in raw.split("\n"):
                line = line.strip().strip('"')

                if (
                    line
                    and "answer" not in line.lower()
                    and len(line.split()) <= 8
                ):
                    variants.append(line)

            queries = [query] + variants[:2]
            print(f"[Expansion] Variants: {queries}")

        except Exception as e:
            print(f"[Expansion] Failed: {e}")
            queries = [query]

        # ─────────────────────────────
        # 3. Retrieval (deduplicated)
        # ─────────────────────────────
        all_docs = []
        seen = set()

        for q in queries:
            docs = self.retrieve(q, config)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        print(f"[Retrieve] {len(all_docs)} unique docs")

        # ─────────────────────────────
        # 4. LLM-based Relevance Grading (NEW)
        # ─────────────────────────────
        top_context = all_docs[0].page_content[:1000] if all_docs else ""

        grader_msg = GRADER_PROMPT.format(
            query=query,
            context=top_context
        )

        try:
            relevance_raw = self.llm._call(grader_msg).lower().strip()

            if "yes" in relevance_raw:
                relevance = "yes"
            else:
                relevance = "no"
            print(f"[Grader] Relevance: {relevance}")
        except Exception as e:
            print(f"[Grader] Failed: {e}")
            relevance = "yes"  # fallback safe default

        # ─────────────────────────────
        # 5. Fallback Logic (NEW)
        # ─────────────────────────────
        if "no" in relevance or not all_docs:
            print("[Fallback] Using general medical knowledge")

            llm_answer = self.llm._call(
                FALLBACK_PROMPT.format(query=query)
            )

            filtered_docs = []
            is_fallback = True

        else:
            # ─────────────────────────────
            # 6. Standard RAG
            # ─────────────────────────────
            # 🔥 REDUCE DOC COUNT (CRITICAL)
            filtered_docs = all_docs[:3]

            context_text = format_docs(filtered_docs)

            # 🔥 HARD LIMIT CONTEXT SIZE
            context_text = truncate_context(context_text, 3500)

            prompt = RAG_PROMPT.format(
                context=context_text,
                query=query
            )

            print(f"[LLM] Generating RAG answer from {len(filtered_docs)} docs...")

            import time
            start = time.time()

            llm_answer = self.llm._call(prompt)

            print(f"[LLM] Time: {round(time.time() - start, 2)}s")

            is_fallback = False

        # ─────────────────────────────
        # 7. Source Attribution
        # ─────────────────────────────
        excerpts = []
        for i, doc in enumerate(filtered_docs, 1):
            book = clean_book_name(doc)
            excerpts.append(
                f"[Source {i} — {book}]\n{doc.page_content[:500].strip()}"
            )

        answer = llm_answer.strip()

        # ─────────────────────────────
        # 8. Return
        # ─────────────────────────────
        return {
            "query": query,
            "query_type": query_type.value,
            "answer": answer,
            "source_docs": filtered_docs,
            "context": top_context,
            "num_docs": len(filtered_docs),
            "fallback_used": is_fallback,
        }

# ─────────────────────────────────────────────
# Pipeline Factory
# ─────────────────────────────────────────────

def build_pipeline(
    endpoint: str = SAGEMAKER_ENDPOINT,
    region:   str = AWS_REGION,
) -> MedicalRAGPipeline:
    """Connect to Pinecone + SageMaker and return ready pipeline."""
    print(f"[Pipeline] Endpoint : {endpoint}")
    print(f"[Pipeline] Region   : {region}")

    vectorstore = load_vectorstore()
    llm         = SageMakerLLM(endpoint_name=endpoint, region_name=region)

    print(f"[Pipeline] ✅ Ready!")
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
        print(f"Query Type : {result['query_type']}")
        print(f"Docs Used  : {result['num_docs']}")
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
                print(f"Sources: {[d.metadata.get('book','?') for d in result['source_docs']]}\n")