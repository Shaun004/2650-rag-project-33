"""LLM generation via Ollama (Sprint: replace Falcon with a free local model)."""

from __future__ import annotations

from typing import Protocol

import requests


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()


class EchoLLM:
    """Deterministic stand-in for tests when Ollama is unavailable."""

    def generate(self, prompt: str) -> str:
        # Extract first document snippet for a grounded stub answer
        marker = "Content:\n"
        if marker in prompt and "Question:" in prompt:
            question = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip()
            content = prompt.split(marker, 1)[1].split("\n\nDocument", 1)[0].strip()
            page_line = ""
            if "Page: p." in prompt:
                page_line = prompt.split("Page: ", 1)[1].split("\n", 1)[0].strip()
            manual = "Samsung TV Manual"
            if "Manual: " in prompt:
                manual = prompt.split("Manual: ", 1)[1].split("\n", 1)[0].strip()
            steps = [line.strip("•- ").strip() for line in content.splitlines() if line.strip()][:5]
            numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
            return (
                f"Based on the manual for your question ({question}):\n\n"
                f"{numbered or content[:400]}\n\n"
                f"Source: {manual}, {page_line or 'page unknown'}."
            )
        return "The manual does not provide enough information."
