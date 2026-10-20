"""Prompt construction for grounded Samsung TV answers."""

from __future__ import annotations

from .retriever import RetrievedChunk

SYSTEM_INSTRUCTIONS = """You are a Samsung TV technical support assistant.

Answer the user's question using ONLY the supplied manual context.

If the context does not contain enough information to answer confidently,
say that the manual does not provide enough information.

Do not invent menu paths or button names.
Do not present general knowledge as if it came from Samsung manuals.
When giving instructions, present them as numbered steps.
Distinguish between different TV models when relevant.
Include the source manual and page number where possible.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page = f"p. {chunk.page}" if chunk.page is not None else "page unknown"
        blocks.append(
            f"Document {i}:\n"
            f"Manual: {chunk.manual}\n"
            f"Model: {chunk.model}\n"
            f"Section: {chunk.section}\n"
            f"Page: {page}\n"
            f"Content:\n{chunk.passage}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
) -> str:
    model_line = f"User TV model: {model}\n" if model else "User TV model: not specified\n"
    if not model:
        model_line += (
            "If menus may differ by model, say so clearly. "
            "If the question is model-specific and the model is unknown, ask which model they have.\n"
        )
    context = format_context(chunks) if chunks else "(no relevant manual passages retrieved)"
    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"{model_line}\n"
        f"Question: {question}\n\n"
        f"{context}\n\n"
        "Answer:"
    )


def insufficient_context_message(model: str | None = None) -> str:
    base = (
        "I could not find enough information in the Samsung TV manuals I have indexed "
        "to answer that confidently. Please try rephrasing, or check the official support page."
    )
    if not model:
        return base + " If you tell me your TV model (for example UA75DU8000WXXY), I can search more precisely."
    return base
