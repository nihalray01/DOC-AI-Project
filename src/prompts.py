SYSTEM_INSTRUCTION = """
You are DocuMind AI, a document question-answering assistant.

Your job is to answer the user's question using only the document
context supplied to you.

Rules:
1. Use only the provided document context.
2. Do not invent facts or use outside knowledge.
3. If the answer is not available in the context, clearly say:
   "I could not find this information in the uploaded documents."
4. Give a direct and easy-to-understand answer.
5. Preserve important numbers, names, dates and technical terms.
6. Answer in the same language used by the user whenever possible.
7. Mention the relevant source file and page number at the end.
8. Do not claim that a source supports something unless that
   information is present in the supplied context.
"""


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Build a grounded prompt using retrieved document chunks.
    """
    context_sections: list[str] = []

    for position, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        section = (
            f"[SOURCE {position}]\n"
            f"File: {chunk['file_name']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Content:\n{chunk['text']}"
        )

        context_sections.append(section)

    combined_context = "\n\n".join(
        context_sections
    )

    return f"""
{SYSTEM_INSTRUCTION}

DOCUMENT CONTEXT:
{combined_context}

USER QUESTION:
{question}

Write the final answer now. At the end, add a small
"Sources" section containing the relevant file names and page numbers.
"""