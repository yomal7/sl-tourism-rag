"""
llm_client.py
-------------
Single entry point for talking to "whichever LLM the .env file says to
use". Everything else in the backend (generate_response.py) calls
`generate(system_prompt, user_prompt)` and doesn't know or care whether
that request ends up going to Gemini or to a local Ollama server.

This exists because the two teammates on this project run different
setups: one has Ollama running locally, the other doesn't and uses the
Gemini API instead. Swapping providers is a one-line change in .env
(LLM_PROVIDER=gemini or LLM_PROVIDER=ollama) — no code change needed.
"""

from backend.config import settings


def generate(system_prompt: str, user_prompt: str) -> str:
    """Routes to the configured provider and returns the model's reply text."""
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        return _generate_gemini(system_prompt, user_prompt)
    elif provider == "ollama":
        return _generate_ollama(system_prompt, user_prompt)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}' in .env — "
            "expected 'gemini' or 'ollama'."
        )


def _generate_gemini(system_prompt: str, user_prompt: str) -> str:
    # Imported lazily so that teammates running Ollama-only never need the
    # google-genai package installed to just run the app... (it's still
    # in pyproject.toml as a shared dependency, but this keeps the failure
    # mode clear if someone strips it out later.)
    from google import genai
    from google.genai import types

    if not settings.gemini_api_key:
        raise RuntimeError(
            "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set in .env. "
            "Get a free key from Google AI Studio and add it to .env."
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text


def _generate_ollama(system_prompt: str, user_prompt: str) -> str:
    import ollama

    client = ollama.Client(host=settings.ollama_host)
    response = client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]
