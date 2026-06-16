"""Shared utilities for orchestration modes."""


def rag_xml(chunks: list) -> str:
    """Format pre-retrieved RAG chunks as an XML block for prompt injection."""
    if not chunks:
        return ""
    parts = [
        f"<chunk path={c.get('source_path', '?')!r} distance={c.get('distance', 0):.4f}>\n"
        f"{c.get('content', '')}\n</chunk>"
        for c in chunks
    ]
    return "<retrieved>\n" + "\n".join(parts) + "\n</retrieved>\n"
