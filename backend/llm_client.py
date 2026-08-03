from backend.config import settings


class LLMServiceError(Exception):

    def __init__(self, user_message: str, retryable: bool = True, status_code: int = 503):
        super().__init__(user_message)
        self.user_message = user_message
        self.retryable = retryable
        self.status_code = status_code


def generate(system_prompt: str, user_prompt: str) -> str:
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        return _generate_gemini(system_prompt, user_prompt)
    elif provider == "ollama":
        return _generate_ollama(system_prompt, user_prompt)
    else:
        raise LLMServiceError(
            f"LLM_PROVIDER in .env is '{settings.llm_provider}' — expected "
            "'gemini' or 'ollama'.",
            retryable=False,
            status_code=500,
        )

# Gemini
def _generate_gemini(system_prompt: str, user_prompt: str) -> str:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors

    if not settings.gemini_api_key:
        raise LLMServiceError(
            "Gemini is selected but GEMINI_API_KEY is missing from .env.",
            retryable=False,
            status_code=500,
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
    except genai_errors.ClientError as e:
        if e.code == 429:
            raise LLMServiceError(
                "Gemini is rate-limiting this API key (too many requests, "
                "or the free-tier quota is used up for now). Wait a few "
                "seconds and try again.",
                retryable=True,
                status_code=429,
            ) from e
        if e.code in (401, 403):
            raise LLMServiceError(
                "Gemini rejected the request — check that GEMINI_API_KEY "
                "in .env is correct and active.",
                retryable=False,
                status_code=500,
            ) from e
        raise LLMServiceError(
            f"Gemini rejected the request ({e.code}).", retryable=False, status_code=502
        ) from e
    except genai_errors.ServerError as e:
        raise LLMServiceError(
            "Gemini's servers are temporarily unavailable. Try again in a moment.",
            retryable=True,
            status_code=503,
        ) from e
    except Exception as e:
        raise LLMServiceError(
            "Couldn't reach Gemini — check your internet connection and try again.",
            retryable=True,
            status_code=503,
        ) from e

    return response.text

# Ollama
def _generate_ollama(system_prompt: str, user_prompt: str) -> str:
    import ollama

    client = ollama.Client(host=settings.ollama_host)
    try:
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except ollama.ResponseError as e:
        raise LLMServiceError(
            f"Ollama rejected the request: {e.error}. Have you run "
            f"`ollama pull {settings.ollama_model}`?",
            retryable=False,
            status_code=500,
        ) from e
    except Exception as e:
        raise LLMServiceError(
            f"Couldn't reach Ollama at {settings.ollama_host}. Is it running? "
            "(`ollama list` should show your models.)",
            retryable=True,
            status_code=503,
        ) from e

    return response["message"]["content"]
