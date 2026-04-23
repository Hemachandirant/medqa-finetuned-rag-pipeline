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
    print(f"[Pinecone]  Connected.")
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
                "temperature":        0.6,   # R1 models work better at 0.6
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
        This is the key trick to get detailed answers from small fine-tuned models.
        """
        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        # Inject prefix into assistant turn so model must continue it
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
                "temperature":        0.6,
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
        QueryType.FACTUAL:     {"k": 3,  "use_mmr": False, "expand_query": False},
        QueryType.PROCEDURAL:  {"k": 5,  "use_mmr": True,  "expand_query": False},
        QueryType.DIAGNOSTIC:  {"k": 7,  "use_mmr": True,  "expand_query": True},
        QueryType.COMPARATIVE: {"k": 8,  "use_mmr": True,  "expand_query": True},
        QueryType.MULTI_HOP:   {"k": 10, "use_mmr": True,  "expand_query": True},
    }
    return configs[query_type]


# ─────────────────────────────────────────────
# Query Expansion
# ─────────────────────────────────────────────

EXPANSION_PROMPT = PromptTemplate.from_template(
    """Generate 3 alternative medical search queries for better document retrieval.
Return only the queries, one per line, no numbering.

Original query: {query}
Alternative queries:"""
)


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

GRADER_PROMPT = PromptTemplate.from_template(
    """Is this document relevant to the medical question? Answer with a score between 0.0 and 1.0.
A score above 0.3 means the document contains useful information about the topic.

Question: {question}
Document excerpt: {document}

Reply in this exact format only:
score: <number between 0.0 and 1.0>"""
)


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
# RAG Prompt
# ─────────────────────────────────────────────

MEDICAL_RAG_PROMPT = PromptTemplate.from_template(
    """Use the following medical textbook excerpts to answer the question.
Explain in detail: include drug names, mechanisms, and clinical reasoning.

Context:
{context}

Question: {question}
Explain in detail."""
)


def format_docs(docs: List[Document]) -> str:
    sections = []
    for i, doc in enumerate(docs, 1):
        book  = doc.metadata.get("book", Path(doc.metadata.get("source", "Unknown")).stem)
        score = doc.metadata.get("relevance_score", "N/A")
        sections.append(f"[Source {i} | {book} | relevance={score}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


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
                    "k":            config["k"],
                    "fetch_k":      config["k"] * 3,
                    "lambda_mult":  0.6,
                },
            )
        else:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": config["k"]}
            )
        return retriever.invoke(query)

    def run(self, query: str) -> Dict:
        # Step 1: Adaptive routing
        query_type = classify_query(query)
        config     = get_retrieval_config(query_type)
        print(f"[Adaptive] Query type: {query_type.value} | k={config['k']} | mmr={config['use_mmr']}")

        # Step 2: Query expansion (only for complex queries — saves LLM round trip)
        queries = [query]
        if config["expand_query"]:
            try:
                queries = expand_medical_query(query, self.llm)
                print(f"[Expansion] Generated {len(queries)} query variants")
            except Exception as e:
                print(f"[Expansion] Skipped: {e}")
                queries = [query]

        # Step 3: Retrieve + deduplicate across all query variants
        all_docs, seen = [], set()
        for q in queries:
            for doc in self.retrieve(q, config):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        print(f"[Retrieve] {len(all_docs)} unique docs retrieved")

        # Step 4: Corrective RAG — grade relevance
        filtered_docs, needs_requery = grade_documents(query, all_docs, self.llm)

        # Step 5: Re-retrieve with expanded k if quality was low
        if needs_requery:
            print("[CRAG] Re-retrieving with expanded k...")
            fallback = self.retrieve(query, {**config, "k": TOP_K_EXPANDED, "use_mmr": True})
            filtered_docs, _ = grade_documents(
                query, fallback, self.llm, RELEVANCE_THRESHOLD * 0.7
            )

        # Fallback: use top raw docs if grader filtered everything
        if not filtered_docs:
            print("[CRAG] Fallback: using top 3 ungraded docs")
            filtered_docs = all_docs[:3]

        # Step 6: Two-stage answer
        # Stage A: Model identifies the key answer (what it's good at)
        # Stage B: Retrieved context provides the clinical explanation
        # This is the correct pattern for small fine-tuned MCQ models
        context = format_docs(filtered_docs)

        # Build answer entirely from retrieved context — more reliable than
        # asking a 1.5B MCQ model to synthesize. Model is used for routing
        # and query expansion; context provides the clinical explanation.
        best_doc     = max(filtered_docs, key=lambda d: d.metadata.get("relevance_score", 0))
        source_book  = best_doc.metadata.get("book", "Medical Textbook")

        # Extract first clean sentence from Source 1 (best prose, not figure captions)
        # Pick the doc with the longest clean first sentence (avoids figure captions)
        def get_first_sentence(doc):
            text = doc.page_content.strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            # Skip sentences that look like figure captions or headers
            for s in sentences:
                if len(s) > 40 and not s.startswith("FIGURE") and not s.startswith("TABLE"):
                    return s
            return sentences[0] if sentences else text[:150]

        # Pick best summary sentence across ALL docs (not just highest score)
        first_sentence = ""
        for doc in filtered_docs:
            candidate = get_first_sentence(doc)
            if len(candidate) > len(first_sentence):
                first_sentence = candidate

        def clean_book_name(doc):
            book = (
                doc.metadata.get("book") or
                doc.metadata.get("source", "") or
                "Medical Textbook"
            )
            # Normalize: strip path separators and .txt
            import re as _re
            book = _re.split(r"[/\\]", book)[-1].replace(".txt", "")
            return book

        # Full answer: summary + all source excerpts with real book names
        excerpts = []
        for i, doc in enumerate(filtered_docs, 1):
            book  = clean_book_name(doc)
            score = doc.metadata.get("relevance_score", "N/A")
            excerpts.append(
                f"[Source {i} — {book} | relevance={score}]\n{doc.page_content[:600].strip()}"
            )

        answer = (
            f"Summary: {first_sentence}\n\n"
            f"Clinical Detail:\n" + "\n\n".join(excerpts)
        )

        return {
            "query":       query,
            "query_type":  query_type.value,
            "answer":      answer,
            "source_docs": filtered_docs,
            "context":     context,
            "num_docs":    len(filtered_docs),
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
        # Interactive mode
        print("\nMedical RAG ready. Type your question (or 'quit' to exit):\n")
        while True:
            q = input("Question: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                result = pipeline_obj.run(q)
                print(f"\nAnswer:\n{result['answer']}\n")
                print(f"Sources: {[d.metadata.get('book','?') for d in result['source_docs']]}\n")