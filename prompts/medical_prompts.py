# System-level persona
SYSTEM_PROMPT = (
    "You are an advanced Medical AI Assistant trained on clinical textbooks and USMLE-style data. "
    "Provide accurate, evidence-based answers with clear clinical reasoning. "
    "Do NOT generate unsupported claims. Be concise but clinically precise."
)
# For Query Expansion (Adaptive RAG)
QUERY_EXPANSION_PROMPT = """
Generate exactly 3 short medical search queries related to the question.

Rules:
- Each query must be a short phrase (NOT a sentence)
- Do NOT include explanations
- Do NOT include words like "answer", "question", or quotes
- One query per line only

Original Query: {query}

Output:
"""

classification_prompt = """
Is the following query medical or not?

Query: {query}

Answer only: medical / non-medical
"""

# The main RAG synthesis prompt
RAG_PROMPT = """
<context>
{context}
</context>

You must answer using ONLY the context.

Step 1: Copy the exact sentence from the context that answers the question.
Step 2: Then explain it clearly in your own words.

STRICT RULES:
- Do NOT change biological mechanisms
- Do NOT introduce new concepts (e.g., receptors, pathways not mentioned)
- If unsure, return only the extracted sentence

Question: {query}

Answer format:

Extracted:
<exact sentence from context>

Explanation:
<your explanation>
"""

# For Corrective RAG (LLM-based Grading)
GRADER_PROMPT = """
Answer ONLY with 'yes' or 'no'.

Is the following context relevant to the medical question?

Question: {query}
Context: {context}
"""

# Fallback Prompt (When RAG fails or context is missing)
FALLBACK_PROMPT = """
The database does not contain enough information for this query.

Provide a general medical overview of: {query}

Structure:
1. Definition
2. Clinical Presentation
3. Diagnosis
4. Treatment

Note: This is general medical knowledge, not from the indexed documents.
"""