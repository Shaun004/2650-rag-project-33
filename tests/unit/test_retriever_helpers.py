from rag.pipeline import _looks_out_of_scope
from rag.retriever import _expand_query, _tokenize


def test_tokenize_wifi_compound():
    tokens = _tokenize("How do I connect Wi-Fi?")
    assert "wifi" in tokens
    assert "wi" not in tokens or "fi" not in tokens


def test_expand_query_wifi():
    expanded = _expand_query("How do I connect Wi-Fi?").lower()
    assert "wireless" in expanded or "network" in expanded


def test_out_of_scope_ceo():
    assert _looks_out_of_scope("Who is the CEO of Samsung?")
    assert not _looks_out_of_scope("How do I connect Wi-Fi?")
