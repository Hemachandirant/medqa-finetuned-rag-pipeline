"""
index_documents.py — Pinecone + Azure OpenAI Embeddings
Replaces ChromaDB with Pinecone for cloud-accessible vector store.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# ─────────────────────────────────────────────
# CONFIG (load from .env)
# ─────────────────────────────────────────────
DATA_PATH     = "./data"
CHUNK_SIZE    = 2500
CHUNK_OVERLAP = 200

# Azure OpenAI
AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT  = os.getenv("AZURE_DEPLOYMENT", "text-embedding-3-small")

# Pinecone
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medqa-textbooks")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")       # aws | gcp | azure
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small = 1536 dims

MEDICAL_SEPARATORS = [
    "\nDiagnosis", "\nTreatment", "\nPathophysiology",
    "\nClinical Features", "\nManagement", "\nEtiology",
    "\nComplications", "\nDefinition", "\nEpidemiology",
    "\nPrognosis", "\n## ", "\n### ", "\n\n", "\n"
]
# ─────────────────────────────────────────────


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        chunk_size=10,
    )


def init_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME in existing:
        print(f"[Pinecone] Index '{PINECONE_INDEX_NAME}' already exists.")
        ans = input("   Delete and rebuild? (y/n): ").strip().lower()
        if ans == "y":
            pc.delete_index(PINECONE_INDEX_NAME)
            print(f"[Pinecone] Deleted index '{PINECONE_INDEX_NAME}'.")
        else:
            print("   Skipping index creation — running verification only...")
            return pc.Index(PINECONE_INDEX_NAME), False

    print(f"[Pinecone] Creating index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        )
    )

    # Wait for index to be ready
    print("[Pinecone] Waiting for index to be ready...")
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(2)

    print(f"[Pinecone] ✅ Index '{PINECONE_INDEX_NAME}' is ready.")
    return pc.Index(PINECONE_INDEX_NAME), True


def load_documents(data_path: str):
    txt_files = list(Path(data_path).glob("**/*.txt"))

    if not txt_files:
        print(f"❌ No .txt files found in {data_path}")
        return []

    print(f"[Load] Found {len(txt_files)} text files:")
    for f in txt_files:
        print(f"   - {f.name}")

    all_docs = []
    for txt_path in txt_files:
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs   = loader.load()
            all_docs.extend(docs)
            print(f"   ✅ Loaded {txt_path.name}")
        except Exception as e:
            print(f"   ⚠️  Failed {txt_path.name}: {e}")

    print(f"\n[Load] Total documents: {len(all_docs)}")
    return all_docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        separators=MEDICAL_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

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

    print(f"[Chunk] {len(docs)} pages → {len(chunks)} chunks")
    print(f"[Chunk] Avg chunk size: {sum(c.metadata['char_count'] for c in chunks) // len(chunks)} chars")
    return chunks


def build_index(chunks):
    print(f"\n[Embed] Using Azure OpenAI {AZURE_DEPLOYMENT}...")
    print(f"[Embed] Indexing {len(chunks)} chunks into Pinecone — estimated ~5-10 minutes...\n")

    embeddings = get_embeddings()

    start = time.time()

    # Upsert in batches to avoid rate limits
    BATCH_SIZE = 100
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"   Upserting batch {i // BATCH_SIZE + 1}/{(len(chunks) // BATCH_SIZE) + 1} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
            )
        else:
            vectorstore.add_documents(batch)

        time.sleep(0.5)  # Respect rate limits

    elapsed = int(time.time() - start)
    print(f"\n✅ Index built!")
    print(f"   Chunks indexed : {len(chunks)}")
    print(f"   Time taken     : {elapsed // 60}m {elapsed % 60}s")
    print(f"   Pinecone index : {PINECONE_INDEX_NAME}")
    return vectorstore


def verify_index():
    print(f"\n[Verify] Testing Pinecone index with sample query...")

    embeddings  = get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    results = vectorstore.similarity_search(
        "treatment for type 2 diabetes mellitus", k=3
    )

    print(f"[Verify] Top 3 results:")
    for i, doc in enumerate(results, 1):
        src = Path(doc.metadata.get("source", "unknown")).name
        print(f"\n   [{i}] {src}  |  section: {doc.metadata.get('section_type', 'N/A')}")
        print(f"       {doc.page_content[:200]}...")

    print(f"\n✅ Pinecone index working correctly!")


if __name__ == "__main__":
    print("=" * 55)
    print("  Medical RAG — Pinecone Indexing")
    print("=" * 55)

    Path(DATA_PATH).mkdir(exist_ok=True)

    _, is_new = init_pinecone_index()

    if not is_new:
        verify_index()
        exit(0)

    docs = load_documents(DATA_PATH)
    if not docs:
        exit(1)

    chunks = chunk_documents(docs)
    build_index(chunks)
    verify_index()

    print("\n🎉 Ready! Now run: python rag_pipeline.py")