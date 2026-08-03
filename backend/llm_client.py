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

# Gemini
def _generate_gemini(system_prompt: str, user_prompt: str) -> str:
    from google import genai
    from google.genai import types

    if not settings.gemini_api_key:
        raise RuntimeError(
            "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set in .env. "
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text

# Ollama
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
