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

# The main RAG synthesis prompt
RAG_PROMPT = """
<context>
{context}
</context>

Answer the clinical question using ONLY the context above.

Question: {query}

Structure your answer:

**Clinical Summary**
(2 sentences max)

**Detailed Analysis**
(Explain mechanism / pathology clearly)

**Clinical Management**
(List treatments or next steps)

**Safety Note**
(Mention contraindications or risks if present)

If the answer is not in the context, say:
"The provided medical database does not contain sufficient information."
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