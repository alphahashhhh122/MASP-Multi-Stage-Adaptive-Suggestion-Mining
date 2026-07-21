"""
llm_backend.py — Unified LLM backend for MASP pipeline.

Uses Ollama HTTP API (not subprocess) for:
  - Text: gemma3:27b-it-qat
  - Vision: gemma3:27b-it-qat (native multimodal — processes actual images)
  - Fast: gemma3:27b-it-qat (same model, lower max_tokens for speed)

All calls go through /api/generate with streaming disabled.
Vision calls send base64 images in the 'images' field — the model ACTUALLY
processes the image pixels, not just a text description.

Change log:
  Uses gemma3:27b-it-qat via Ollama HTTP API for all text and vision calls.
"""

import json
import re
import time
import logging
import requests

logger = logging.getLogger(__name__)

_config = {
    "provider": "ollama",
    "ollama_base_url": "http://localhost:11434",
    "ollama_text_model": "gemma3:27b-it-qat",
    "ollama_vision_model": "gemma3:27b-it-qat",
    "ollama_fast_model": "gemma3:27b-it-qat",
    "max_retries": 3,
    "temperature": 0.0,
    "max_tokens": 4096,
    "retry_delay": 2.0,
    "timeout": 180,
}


def get_config():
    return dict(_config)


def set_config(**kw):
    _config.update(kw)


def _ollama_generate(model, prompt, images=None):
    """Call Ollama HTTP API /api/generate.

    Args:
        model: Model name (e.g., "gemma3:27b-it-qat")
        prompt: Full prompt text (system + user combined)
        images: Optional list of base64-encoded image strings (for vision)

    Returns:
        Raw response text from the model.
    """
    url = f"{_config['ollama_base_url']}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": _config["temperature"],
            "num_predict": _config["max_tokens"],
        },
    }

    # Add images for vision calls — Ollama expects list of base64 strings
    if images:
        valid_images = [img for img in images if img and len(img) > 100]
        if valid_images:
            payload["images"] = valid_images

    for attempt in range(_config["max_retries"]):
        try:
            resp = requests.post(url, json=payload, timeout=_config["timeout"])
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

        except requests.exceptions.Timeout:
            logger.warning(f"Ollama timeout (attempt {attempt+1}/{_config['max_retries']})")
            if attempt == _config["max_retries"] - 1:
                raise
            time.sleep(_config["retry_delay"] * (attempt + 1))

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {_config['ollama_base_url']}")
            if attempt == _config["max_retries"] - 1:
                raise
            time.sleep(_config["retry_delay"] * (attempt + 1))

        except Exception as e:
            logger.warning(f"Ollama error (attempt {attempt+1}): {e}")
            if attempt == _config["max_retries"] - 1:
                raise
            time.sleep(_config["retry_delay"])

    return ""


def _parse_json(raw):
    """Extract JSON object from LLM response, handling markdown fences and preamble."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Try full response first
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fallback: find balanced braces for first JSON object
    depth = 0
    start = None
    for i, c in enumerate(raw):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    pass
                break
    # Last resort: try to extract individual suggestion objects from array
    suggestions = []
    for m in re.finditer(r'\{"text":\s*"[^"]+",\s*"confidence":[^}]+\}', raw):
        try:
            suggestions.append(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
    if suggestions:
        return {"suggestions": suggestions}
    raise ValueError(f"LLM returned invalid JSON\nRaw response: {raw[:500]}")


def call_llm(system, user, fast=False):
    """Call text-only LLM. Returns (parsed_json, message_history)."""
    model = _config["ollama_fast_model"] if fast else _config["ollama_text_model"]
    prompt = f"{system}\n\n{user}"

    raw = _ollama_generate(model, prompt)
    parsed = _parse_json(raw)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": raw},
    ]
    return parsed, messages


def call_llm_vision(system, user_text, image_b64, mime="image/jpeg"):
    """Call vision LLM with an actual image.

    The image is sent as base64 via Ollama's 'images' field.
    gemma3:27b-it-qat has native multimodal capability and will
    actually process the image pixels.

    Args:
        system: System prompt
        user_text: User prompt text
        image_b64: Base64-encoded image data
        mime: MIME type (for logging)

    Returns:
        Tuple of (parsed_json_dict, message_history)
    """
    model = _config["ollama_vision_model"]
    prompt = f"{system}\n\n{user_text}"

    # Send image to the model — REAL vision processing
    images = [image_b64] if image_b64 and len(image_b64) > 100 else None

    if images:
        logger.info(f"[vision] Sending actual image ({len(image_b64)//1024}KB) to {model}")
    else:
        logger.info(f"[vision] No image data — text-only fallback")

    raw = _ollama_generate(model, prompt, images=images)
    parsed = _parse_json(raw)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user_text}\n[Image: {len(image_b64)//1024 if image_b64 else 0}KB {mime}]" if images else user_text},
        {"role": "assistant", "content": raw},
    ]
    return parsed, messages


def check_ollama():
    """Verify Ollama is running and required models are available."""
    try:
        resp = requests.get(f"{_config['ollama_base_url']}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]

        print("Ollama running. Models available:")
        for m in models:
            print(f"  - {m}")

        needed = _config["ollama_text_model"]
        found = any(needed in m or m in needed for m in models)

        if found:
            print(f"\n✓ Required model '{needed}' is available")
            return True
        else:
            print(f"\n✗ Required model '{needed}' NOT found")
            print(f"  Install with: ollama pull {needed}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Ollama at {_config['ollama_base_url']}")
        print("  Start with: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Ollama check failed: {e}")
        return False


if __name__ == "__main__":
    check_ollama()
