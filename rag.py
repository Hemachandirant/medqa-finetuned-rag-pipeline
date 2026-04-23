"""
Medical RAG Pipeline
====================
Strategy: Hybrid Adaptive + Corrective RAG
- Adaptive RAG: Routes queries to different retrieval strategies based on query complexity
- Corrective RAG: Validates retrieved documents and re-queries if relevance is low
- Chunking: Semantic chunking (respects medical concept boundaries)
- Embeddings: BioBERT-based (domain-specific medical embeddings)
- LLM: Calls deployed AWS SageMaker endpoint (no local GPU needed)
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

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field


# ─────────────────────────────────────────────
# Configuration (load from .env)
# ─────────────────────────────────────────────

SAGEMAKER_ENDPOINT = os.getenv("ENDPOINT_NAME", "medqa-deepseek-v2")
AWS_REGION         = os.getenv("AWS_REGION", "ap-south-1")
EMBED_MODEL = "all-MiniLM-L6-v2"  # Public, open-source embeddings model
CHROMA_PATH        = "./chroma_db"
DATA_PATH          = "./data"
CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 64
TOP_K_INITIAL      = 5
TOP_K_EXPANDED     = 10
RELEVANCE_THRESHOLD = 0.35

SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant trained on clinical textbooks. "
    "Answer the question using ONLY the provided context. If the context does not "
    "contain sufficient information, state that clearly — do not hallucinate. "
    "Always use precise medical terminology."
)


# ─────────────────────────────────────────────
# SageMaker LLM Wrapper (replaces local loading)
# ─────────────────────────────────────────────

class SageMakerLLM(BaseLLM):
    """
    LangChain-compatible LLM that calls your deployed
    SageMaker endpoint instead of loading model locally.
    """
    endpoint_name: str = Field(default=SAGEMAKER_ENDPOINT)
    region_name:   str = Field(default=AWS_REGION)
    max_new_tokens: int = Field(default=512)

    @property
    def _llm_type(self) -> str:
        return "sagemaker"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        runtime = boto3.client("sagemaker-runtime", region_name=self.region_name)

        # Qwen2 chat template
        formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        payload = {
            "inputs": formatted,
            "parameters": {
                "max_new_tokens":     self.max_new_tokens,
                "temperature":        0.1,
                "repetition_penalty": 1.1,
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
            return text.strip()
        except Exception as e:
            print(f"[SageMaker] Error calling endpoint: {e}")
            return "Error: Could not get response from model endpoint."

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import LLMResult, Generation
        generations = [[Generation(text=self._call(p, stop, run_manager))] for p in prompts]
        return LLMResult(generations=generations)


# ─────────────────────────────────────────────
# Query Complexity Classifier (Adaptive RAG)
# ─────────────────────────────────────────────

class QueryType(Enum):
    FACTUAL    = "factual"
    PROCEDURAL = "procedural"
    DIAGNOSTIC = "diagnostic"
    COMPARATIVE= "comparative"
    MULTI_HOP  = "multi_hop"


QUERY_PATTERNS = {
    QueryType.FACTUAL:    [r"\bwhat is\b", r"\bdefine\b", r"\bnormal range\b", r"\bvalue of\b"],
    QueryType.PROCEDURAL: [r"\bhow (is|to|do)\b", r"\bprocedure\b", r"\bsteps\b", r"\bperform\b"],
    QueryType.DIAGNOSTIC: [r"\bdifferential\b", r"\bdiagnose\b", r"\bpresent(s|ing)?\b", r"\bsymptom\b"],
    QueryType.COMPARATIVE:[r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bdifference\b"],
    QueryType.MULTI_HOP:  [r"\baffect\b", r"\brelationship\b", r"\bimpact\b", r"\binteraction\b"],
}


def classify_query(query: str) -> QueryType:
    q = query.lower()
    for qtype, patterns in QUERY_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            return qtype
    return QueryType.FACTUAL


def get_retrieval_config(query_type: QueryType) -> Dict:
    configs = {
        QueryType.FACTUAL:    {"k": 3,  "use_mmr": False, "expand_query": False},
        QueryType.PROCEDURAL: {"k": 5,  "use_mmr": True,  "expand_query": False},
        QueryType.DIAGNOSTIC: {"k": 7,  "use_mmr": True,  "expand_query": True},
        QueryType.COMPARATIVE:{"k": 8,  "use_mmr": True,  "expand_query": True},
        QueryType.MULTI_HOP:  {"k": 10, "use_mmr": True,  "expand_query": True},
    }
    return configs[query_type]


# ─────────────────────────────────────────────
# Medical Semantic Chunker
# ─────────────────────────────────────────────

MEDICAL_SECTION_HEADERS = [
    "\n## ", "\n### ", "\nDiagnosis", "\nTreatment", "\nPathophysiology",
    "\nClinical Features", "\nManagement", "\nEtiology", "\nComplications",
    "\nDefinition", "\nEpidemiology", "\nPrognosis",
]


def build_medical_splitter() -> RecursiveCharacterTextSplitter:
    separators = MEDICAL_SECTION_HEADERS + ["\n\n", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )


# ─────────────────────────────────────────────
# Document Ingestion — supports .txt and .pdf
# ─────────────────────────────────────────────

def load_and_chunk_documents(data_path: str) -> List[Document]:
    """Load .txt (and .pdf if present) medical textbooks."""
    all_raw = []

    # Load .txt files (MedQA textbooks)
    txt_files = list(Path(data_path).glob("**/*.txt"))
    for txt_path in txt_files:
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs   = loader.load()
            # Tag source book name from filename
            for doc in docs:
                doc.metadata["book"] = txt_path.stem
            all_raw.extend(docs)
            print(f"   ✅ Loaded {txt_path.name} ({len(docs)} docs)")
        except Exception as e:
            print(f"   ⚠️  Failed {txt_path.name}: {e}")

    # Also load .pdf files if any exist
    pdf_files = list(Path(data_path).glob("**/*.pdf"))
    for pdf_path in pdf_files:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(pdf_path))
            docs   = loader.load()
            for doc in docs:
                doc.metadata["book"] = pdf_path.stem
            all_raw.extend(docs)
            print(f"   ✅ Loaded {pdf_path.name} ({len(docs)} pages)")
        except Exception as e:
            print(f"   ⚠️  Failed {pdf_path.name}: {e}")

    if not all_raw:
        raise ValueError(f"No documents found in {data_path}. Add .txt or .pdf files.")

    splitter = build_medical_splitter()
    chunks   = splitter.split_documents(all_raw)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]   = i
        chunk.metadata["char_count"] = len(chunk.page_content)
        content_lower = chunk.page_content.lower()
        for section in ["diagnosis", "treatment", "pathophysiology", "management", "etiology"]:
            if section in content_lower:
                chunk.metadata["section_type"] = section
                break
        else:
            chunk.metadata["section_type"] = "general"

    print(f"\n[Ingestion] {len(all_raw)} docs → {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: List[Document], persist_path: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectorstore.persist()
    print(f"[VectorStore] Indexed {len(chunks)} chunks → {persist_path}")
    return vectorstore


def load_vectorstore(persist_path: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
    )


# ─────────────────────────────────────────────
# Query Expansion
# ─────────────────────────────────────────────

EXPANSION_PROMPT = PromptTemplate.from_template(
    """Generate 3 alternative medical search queries for better document retrieval.
Return only the queries, one per line, no numbering.

Original query: {query}
Alternative queries:"""
)


def expand_medical_query(query: str, llm) -> List[str]:
    chain   = EXPANSION_PROMPT | llm | StrOutputParser()
    result  = chain.invoke({"query": query})
    variants = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return [query] + variants[:3]


# ─────────────────────────────────────────────
# Corrective RAG — Relevance Grader
# ─────────────────────────────────────────────

GRADER_PROMPT = PromptTemplate.from_template(
    """Score the relevance of this document to the medical question.
Return ONLY JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}

Question: {question}
Document: {document}"""
)


def grade_documents(
    query: str, docs: List[Document], llm, threshold: float = RELEVANCE_THRESHOLD
) -> Tuple[List[Document], bool]:
    grader_chain = GRADER_PROMPT | llm | StrOutputParser()
    filtered, scores = [], []

    for doc in docs:
        try:
            result     = grader_chain.invoke({"question": query, "document": doc.page_content[:800]})
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            score      = float(json.loads(json_match.group()).get("score", 0)) if json_match else 0.0
        except Exception:
            score = 0.0

        scores.append(score)
        if score >= threshold:
            doc.metadata["relevance_score"] = score
            filtered.append(doc)

    needs_requery = len(filtered) < 2 or (scores and max(scores) < threshold)
    print(f"[CRAG] {len(filtered)}/{len(docs)} docs passed grading. Re-query: {needs_requery}")
    return filtered, needs_requery


# ─────────────────────────────────────────────
# RAG Prompt
# ─────────────────────────────────────────────

MEDICAL_RAG_PROMPT = PromptTemplate.from_template(
    """You are a knowledgeable medical assistant trained on clinical textbooks.
Answer the question using ONLY the provided context. If insufficient, say so clearly.

Always:
- Cite specific findings from the context
- Use precise medical terminology
- Flag if clinical judgement beyond textbooks is needed

Context:
{context}

Question: {question}

Answer:"""
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
    def __init__(self, vectorstore: Chroma, llm: SageMakerLLM):
        self.vectorstore = vectorstore
        self.llm         = llm

    def retrieve(self, query: str, config: Dict) -> List[Document]:
        if config["use_mmr"]:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": config["k"], "fetch_k": config["k"] * 3, "lambda_mult": 0.6},
            )
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": config["k"]})
        return retriever.invoke(query)

    def run(self, query: str) -> Dict:
        # Step 1: Adaptive routing
        query_type = classify_query(query)
        config     = get_retrieval_config(query_type)
        print(f"[Adaptive] Query type: {query_type.value} | k={config['k']} mmr={config['use_mmr']}")

        # Step 2: Query expansion
        queries = [query]
        if config["expand_query"]:
            queries = expand_medical_query(query, self.llm)
            print(f"[Expansion] {len(queries)} queries")

        # Step 3: Retrieve + deduplicate
        all_docs, seen = [], set()
        for q in queries:
            for doc in self.retrieve(q, config):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        # Step 4: Corrective RAG grading
        filtered_docs, needs_requery = grade_documents(query, all_docs, self.llm)

        # Step 5: Re-retrieve if needed
        if needs_requery:
            print("[CRAG] Re-retrieving with expanded k...")
            fallback = self.retrieve(query, {**config, "k": TOP_K_EXPANDED, "use_mmr": True})
            filtered_docs, _ = grade_documents(query, fallback, self.llm, RELEVANCE_THRESHOLD * 0.7)

        if not filtered_docs:
            filtered_docs = all_docs[:3]

        # Step 6: Generate answer via SageMaker
        context = format_docs(filtered_docs)
        chain   = MEDICAL_RAG_PROMPT | self.llm | StrOutputParser()
        answer  = chain.invoke({"context": context, "question": query})

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
    data_path:    str  = DATA_PATH,
    chroma_path:  str  = CHROMA_PATH,
    endpoint:     str  = SAGEMAKER_ENDPOINT,
    region:       str  = AWS_REGION,
    rebuild_index: bool = False,
) -> MedicalRAGPipeline:
    """Build or reload the full RAG pipeline."""
    print(f"[Pipeline] SageMaker endpoint: {endpoint} ({region})")

    if rebuild_index or not Path(chroma_path).exists():
        print(f"[Pipeline] Building index from {data_path}...")
        chunks      = load_and_chunk_documents(data_path)
        vectorstore = build_vectorstore(chunks, chroma_path)
    else:
        print(f"[Pipeline] Loading existing index from {chroma_path}...")
        vectorstore = load_vectorstore(chroma_path)

    llm = SageMakerLLM(endpoint_name=endpoint, region_name=region)
    print(f"[Pipeline] ✅ Ready!")
    return MedicalRAGPipeline(vectorstore=vectorstore, llm=llm)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical RAG Pipeline")
    parser.add_argument("--query",    type=str, help="Medical question")
    parser.add_argument("--rebuild",  action="store_true", help="Re-index documents")
    parser.add_argument("--endpoint", type=str, default=SAGEMAKER_ENDPOINT)
    parser.add_argument("--region",   type=str, default=AWS_REGION)
    args = parser.parse_args()

    pipeline_obj = build_pipeline(
        endpoint=args.endpoint,
        region=args.region,
        rebuild_index=args.rebuild,
    )

    if args.query:
        result = pipeline_obj.run(args.query)
        print("\n" + "="*60)
        print(f"Query Type : {result['query_type']}")
        print(f"Docs Used  : {result['num_docs']}")
        print(f"\nAnswer:\n{result['answer']}")
    else:
        print("Pipeline ready. Use --query 'your medical question' to run.")