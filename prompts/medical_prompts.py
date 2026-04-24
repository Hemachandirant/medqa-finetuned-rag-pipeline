"""
medical_prompts.py
==================
All prompt templates for the Medical RAG Pipeline.

Changes vs original
-------------------
- Removed dead `classification_prompt` (was defined but never imported/used)
- Added citation instruction to RAG_PROMPT_FACTUAL (was missing; other prompts had it)
- Added RAG_PROMPT_PROCEDURAL — step-by-step structure (PROCEDURAL was wrongly reusing MECHANISTIC)
- Added RAG_PROMPT_COMPARATIVE — comparison structure (COMPARATIVE was wrongly reusing MECHANISTIC)
- QUERY_EXPANSION_PROMPT now says "exactly 2" to match the [:2] slice in expand_medical_query()
- GRADER_PROMPT now includes a negative example so the model has both reference directions
- FALLBACK_PROMPT now handles procedural / lab / dosing queries, not just pharma and disease

P01  All structured templates rewritten from numbered outline format to prose instructions.
     The fine-tuned DeepSeek was echoing numbered section headers verbatim (e.g. the answer
     to "How is a lumbar puncture performed?" was literally "1. Indication — when this
     procedure is performed.") because it had not been trained on that template style.
     Prose instructions describing the desired output shape are more robust across fine-tunes.
"""


# ─────────────────────────────────────────────
# System persona
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an advanced Medical AI Assistant trained on clinical textbooks and USMLE-style data. "
    "Provide accurate, evidence-based answers with clear clinical reasoning. "
    "Do NOT generate unsupported claims. Be concise but clinically precise."
)


# ─────────────────────────────────────────────
# Query expansion
# FIX: changed "exactly 3" → "exactly 2" to match the [:2] slice used in
#      expand_medical_query(); previously the prompt produced one unused variant.
# ─────────────────────────────────────────────

QUERY_EXPANSION_PROMPT = """
Generate exactly 2 short medical search queries related to the question.

Rules:
- Each query must be a short phrase (NOT a full sentence)
- Do NOT include explanations, preamble, or quotes
- Do NOT use words like "answer" or "question"
- One query per line only

Original Query: {query}

Output:
"""


# ─────────────────────────────────────────────
# RAG synthesis prompts
# ─────────────────────────────────────────────

# FIX: added citation instruction — was missing while the other prompts had it.
RAG_PROMPT_FACTUAL = """
<context>
{context}
</context>

Answer the question using ONLY the context above.
- Extract the most relevant sentence(s) directly.
- If the answer is a value or definition, state it plainly.
- Do not add information not present in the context.
- After each statement add a citation marker like [Source 1] or [Source 2]
  matching the source excerpts above. Only cite sources that directly support
  the statement.

Question: {query}

Answer:
"""

RAG_PROMPT_MECHANISTIC = """
<context>
{context}
</context>

Using ONLY the context above, write a mechanistic explanation that covers three things in order.
First, describe the primary mechanism at the molecular or cellular level. Second, explain the
downstream physiological consequences of that mechanism. Third, state what this means clinically
for the patient. After each individual statement add a citation like [Source N] that matches one
of the source excerpts above. Do not introduce pathways, receptors, or drugs not mentioned in
the context.

Question: {query}

Answer:
"""

# P01: rewritten as prose instructions.
# The previous numbered outline ("1. Indication — when this procedure is performed")
# was echoed verbatim as the answer by the fine-tuned model, which had not seen that
# template format during training. Prose instructions are safer across fine-tune variants.
RAG_PROMPT_PROCEDURAL = """
<context>
{context}
</context>

Using ONLY the context above, answer the question about this clinical procedure.

Begin by stating when the procedure is indicated. Then describe how the patient is positioned
and how the site is prepared. Walk through the technique step by step in the order described
in the context. Finally, mention any contraindications, complications, or safety precautions
that appear in the context. After each point add a citation like [Source N] matching the
excerpt it comes from. Do not add steps, equipment, or warnings not present in the context.

Question: {query}

Answer:
"""

RAG_PROMPT_DIAGNOSTIC = """
<context>
{context}
</context>

Using ONLY the context above, answer the diagnostic question. Describe the key clinical and
investigative findings that point toward the diagnosis. If the context mentions differentials,
include them. Then state the most appropriate next diagnostic step according to the context.
After each point add a citation like [Source N] matching the excerpt it comes from.

Question: {query}

Answer:
"""

# P01: same prose-instruction rewrite as PROCEDURAL and MECHANISTIC.
RAG_PROMPT_COMPARATIVE = """
<context>
{context}
</context>

Using ONLY the context above, compare the items asked about in the question. Briefly describe
each item individually, then explain what they have in common. Then cover the key differences —
including mechanism, indication, side-effect profile, and any other axes that appear in the
context. If the context states which is preferred and when, end with that clinical bottom line.
After each point add a citation like [Source N] matching the excerpt it comes from. Do not
introduce comparisons not supported by the context.

Question: {query}

Answer:
"""


# ─────────────────────────────────────────────
# Corrective RAG — LLM relevance grader
# FIX: added a negative example so the model has both reference directions
#      and is less biased toward "yes" for non-pharmacology queries.
# ─────────────────────────────────────────────

GRADER_PROMPT = """
Answer ONLY with the single word 'yes' or 'no'. No punctuation, no explanation.

Example 1 — relevant:
Question: What is the mechanism of metformin?
Context: Metformin reduces hepatic glucose output by antagonizing glucagon signalling.
Answer: yes

Example 2 — not relevant:
Question: What are the symptoms of appendicitis?
Context: The mitral valve controls blood flow between the left atrium and left ventricle.
Answer: no

Now answer:
Question: {query}
Context: {context}
Answer:"""


# ─────────────────────────────────────────────
# Fallback — parametric knowledge only
# FIX: added branches for procedural, lab/interpretation, and dosing queries
#      so the model doesn't force everything into pharma or disease templates.
# ─────────────────────────────────────────────

FALLBACK_PROMPT = """
The indexed documents do not contain sufficient information for this query.
Provide a thorough medical overview of: {query}

If this is a PHARMACOLOGY or MECHANISM question, begin by naming the drug class and its
primary molecular target. Then explain the mechanism of action and resulting physiological
effects. Close with clinical use and key adverse effects.

If this is a DISEASE or DIAGNOSIS question, open with a definition and brief epidemiology.
Describe the typical clinical presentation, then the key diagnostic tests and findings.
End with the standard treatment approach.

If this is a PROCEDURE question, state the indication first. Describe patient preparation
and required equipment. Walk through the technique in the order it is performed. Finish
with known complications and contraindications.

If this is a LAB or INTERPRETATION question, explain what the test measures and its normal
reference range. Describe the clinical significance of high and low values. List common
causes of abnormal results.

If this is a DOSING or PHARMACOKINETICS question, state the standard dosing regimen and
route of administration. Cover bioavailability, half-life, and any renal or hepatic dose
adjustments. End with key monitoring parameters.

At the end of every answer, state clearly:
"This answer is based on general medical knowledge, not retrieved source documents."
"""