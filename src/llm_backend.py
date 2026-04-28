"""
LLM inference backends for the wargame simulation.

Supports two backends:
  - OllamaBackend: local inference via Ollama REST API (used for main experiments)
  - GroqBackend: cloud inference via Groq API (optional, for larger models)

Both return LLMResponse objects with parsed action JSON + usage stats.
"""

import json
import re
import time
import requests
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any backend."""
    raw_text: str
    parsed: Optional[Dict]
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    model: str


class OllamaBackend:
    """Local inference through Ollama's chat API."""

    def __init__(self, model_id: str, base_url: str = "http://localhost:11434"):
        self.model_id = model_id
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed

        start = time.time()
        try:
            resp = requests.post(self.api_url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                raw_text=f"ERROR: {e}", parsed=None,
                prompt_tokens=0, completion_tokens=0,
                latency_sec=time.time() - start, model=self.model_id,
            )
        latency = time.time() - start

        data = resp.json()
        raw_text = data.get("message", {}).get("content", "")
        parsed = _parse_action_json(raw_text)

        return LLMResponse(
            raw_text=raw_text,
            parsed=parsed,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_sec=latency,
            model=self.model_id,
        )


class GroqBackend:
    """Cloud inference via Groq API (OpenAI-compatible endpoint)."""

    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        start = time.time()
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                raw_text=f"ERROR: {e}", parsed=None,
                prompt_tokens=0, completion_tokens=0,
                latency_sec=time.time() - start, model=self.model_id,
            )
        latency = time.time() - start

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        raw_text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        parsed = _parse_action_json(raw_text)

        return LLMResponse(
            raw_text=raw_text,
            parsed=parsed,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_sec=latency,
            model=self.model_id,
        )


def _parse_action_json(text: str) -> Optional[Dict]:
    """
    Extract the action JSON from LLM output.

    Small models often wrap JSON in markdown fences, include preamble text,
    or produce slightly malformed output. This parser tries multiple
    strategies in order of reliability.
    """
    text = text.strip()

    # Strategy 1: markdown code fences (```json ... ```)
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Strategy 2: find outermost { ... } in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: regex extraction of individual fields
    # (last resort when JSON is too broken to parse whole)
    match = re.search(r'"action_number"\s*:\s*(\d+)', text)
    if match:
        action_num = int(match.group(1))
        name_match = re.search(r'"action_name"\s*:\s*"([^"]+)"', text)
        target_match = re.search(r'"target_nation"\s*:\s*"([^"]*)"', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        return {
            "action_number": action_num,
            "action_name": name_match.group(1) if name_match else f"Action_{action_num}",
            "target_nation": target_match.group(1) if target_match else None,
            "reasoning": reasoning_match.group(1) if reasoning_match else "",
        }

    return None


def get_backend(
    model_key: str,
    ollama_url: str = "http://localhost:11434",
    groq_api_key: Optional[str] = None,
):
    """Factory function: returns the right backend for a given model key."""
    from src.config import MODELS, GROQ_MODELS

    if model_key in MODELS:
        return OllamaBackend(MODELS[model_key].ollama_id, ollama_url)
    elif model_key in GROQ_MODELS and groq_api_key:
        return GroqBackend(GROQ_MODELS[model_key].ollama_id, groq_api_key)
    else:
        raise ValueError(f"Unknown model key: {model_key}")
