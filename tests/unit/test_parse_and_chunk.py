from ingestion.chunk_documents import chunk_manual_text
from ingestion.parse_documents import clean_text, parse_manual_text, parse_page_span


SAMPLE = """
# Guide

## Connections

### Connecting to the network  (p. 9-12)

You can get access to the network through your TV.
Home > Settings All Settings Connection Network Open Network Settings
Establishing a wireless network connection
Select Wireless and enter the password.

### Connecting an Antenna (Aerial)  (p. 8)

You can connect an antenna cable to your TV.
"""


def test_parse_page_span():
    title, start, end = parse_page_span("Connecting to the network  (p. 9-12)")
    assert title == "Connecting to the network"
    assert start == 9
    assert end == 12


def test_clean_removes_noise():
    dirty = "Hello\nPage 3\nUser Manual\nWorld\n"
    cleaned = clean_text(dirty)
    assert "Page 3" not in cleaned
    assert "User Manual" not in cleaned
    assert "Hello" in cleaned
    assert "World" in cleaned


def test_section_chunking_keeps_wifi_together():
    sections = parse_manual_text(SAMPLE)
    chunks = chunk_manual_text(sections, strategy="section")
    assert len(chunks) >= 2
    wifi = next(c for c in chunks if "network" in c.section.lower())
    assert "Wireless" in wifi.text
    assert wifi.page_start == 9


def test_fixed_and_sentence_strategies_produce_chunks():
    sections = parse_manual_text(SAMPLE)
    assert chunk_manual_text(sections, strategy="fixed")
    assert chunk_manual_text(sections, strategy="sentence")
