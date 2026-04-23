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
RAG_PROMPT_FACTUAL = """
<context>
{context}
</context>

Answer the question using ONLY the context above.
- Extract the most relevant sentence(s) directly
- If the answer is a value or definition, state it plainly
- Do not add information not present in the context

Question: {query}

Answer:
"""

RAG_PROMPT_MECHANISTIC = """
<context>
{context}
</context>

Using ONLY the context above, explain the mechanism in full.

Structure your answer as:
1. Primary mechanism (molecular/cellular level)
2. Downstream effects (physiological consequences)
3. Clinical relevance (what this means for the patient)

After each statement, add a citation like [Source 1] matching the context.
Do not introduce pathways or receptors not mentioned in the context.

Question: {query}

Answer:
"""

RAG_PROMPT_DIAGNOSTIC = """
<context>
{context}
</context>

Using ONLY the context above, answer the diagnostic question.

Structure:
1. Key findings that support the diagnosis
2. Differentials to consider (only if mentioned in context)
3. Next diagnostic step

Cite each point as [Source N].

Question: {query}

Answer:
"""

# For Corrective RAG (LLM-based Grading)
GRADER_PROMPT = """
Answer ONLY with the single word 'yes' or 'no'. No punctuation, no explanation.

Example:
Question: What is the mechanism of metformin?
Context: Metformin reduces hepatic glucose output by antagonizing glucagon signalling.
Answer: yes

Now answer:
Question: {query}
Context: {context}
Answer:"""

# Fallback Prompt (When RAG fails or context is missing)
FALLBACK_PROMPT = """
The indexed documents do not contain sufficient information for this query.

Provide a thorough medical overview of: {query}

If this is a pharmacology/mechanism question, structure as:
1. Drug class and primary target
2. Molecular mechanism of action
3. Physiological effects
4. Clinical use and key side effects

If this is a disease/diagnosis question, structure as:
1. Definition and epidemiology
2. Clinical Presentation
3. Diagnosis
4. Treatment

Clearly state at the end: "This answer is based on general medical knowledge."
"""