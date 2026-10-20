from ingestion.build_collection import build_collection_rows, chunk_id_for
from ingestion.chunk_documents import Chunk
from rag.prompts import build_prompt, insufficient_context_message
from rag.retriever import RetrievedChunk, model_matches


def test_model_matches_short_and_full():
    assert model_matches("DU8000", "UA75DU8000WXXY", "Crystal UHD DU8000")
    assert model_matches("UA75DU8000WXXY", "UA75DU8000WXXY", "")
    assert not model_matches("QN90D", "UA75DU8000WXXY", "Crystal UHD DU8000")


def test_chunk_id_and_collection_metadata():
    chunk = Chunk(
        text="hello",
        chapter="Connections",
        section="Wi-Fi",
        page_start=9,
        page_end=12,
        chunk_index=3,
    )
    cid = chunk_id_for("UA75DU8000WXXY", chunk)
    assert "p9" in cid
    rows = build_collection_rows(
        [chunk],
        {
            "model_number": "UA75DU8000WXXY",
            "model_family": "Crystal UHD DU8000",
            "manual_title": "Samsung DU8000 Manual",
            "document_url": "https://example.com",
            "language": "English",
        },
    )
    assert rows[0]["page_number"] == 9
    assert rows[0]["section"] == "Wi-Fi"
    assert rows[0]["manufacturer"] == "Samsung"


def test_prompt_requires_context_and_citations_instruction():
    chunks = [
        RetrievedChunk(
            passage_id="x",
            passage="Open Network Settings Wireless",
            score=0.9,
            manual="Samsung DU8000 Manual",
            model="UA75DU8000WXXY",
            page=9,
            section="Connecting to the network",
            chapter="Connections",
            document_url="",
        )
    ]
    prompt = build_prompt("How do I connect Wi-Fi?", chunks, model="UA75DU8000WXXY")
    assert "ONLY the supplied manual context" in prompt
    assert "Document 1" in prompt
    assert "p. 9" in prompt


def test_insufficient_context_asks_for_model_when_missing():
    msg = insufficient_context_message(None)
    assert "model" in msg.lower()
