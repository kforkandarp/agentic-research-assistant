"""
src/llm.py
Unified LLM instantiation and resilient invocation wrapper.
Uses two Groq API keys with automatic failover (Groq Key 1 -> Groq Key 2)
to double rate-limit quotas while keeping the model (LLaMA 3.3 70B) identical.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError, APIError

load_dotenv()
logger = logging.getLogger("LLMFactory")

MODEL_NAME = "openai/gpt-oss-120b"
FAST_MODEL_NAME = "openai/gpt-oss-20b"


def get_llm(temperature: float = 0.0):
    """
    Constructs primary LLaMA 3.3 70B client using GROQ_API_KEY1.
    If GROQ_API_KEY2 is present, attaches it as an automatic fallback LLM instance.
    """
    primary_key = os.getenv("GROQ_API_KEY1") or os.getenv("GROQ_API_KEY")
    secondary_key = os.getenv("GROQ_API_KEY2")

    primary_llm = ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=primary_key,
    )

    # Secondary Groq key fallback instance (same model, second account quota)
    if secondary_key:
        fallback_llm = ChatGroq(
            model=MODEL_NAME,
            temperature=temperature,
            api_key=secondary_key,
        )
        return primary_llm.with_fallbacks([fallback_llm])

    return primary_llm


def get_fast_llm(temperature: float = 0.0):
    """
    Constructs fast router LLaMA 3.1 8B client with Key 1 -> Key 2 fallback.
    """
    primary_key = os.getenv("GROQ_API_KEY1") or os.getenv("GROQ_API_KEY")
    secondary_key = os.getenv("GROQ_API_KEY2")

    primary_llm = ChatGroq(
        model=FAST_MODEL_NAME,
        temperature=temperature,
        api_key=primary_key,
    )

    if secondary_key:
        fallback_llm = ChatGroq(
            model=FAST_MODEL_NAME,
            temperature=temperature,
            api_key=secondary_key,
        )
        return primary_llm.with_fallbacks([fallback_llm])

    return primary_llm


@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def invoke_with_retry(llm, messages):
    """
    Wraps LLM invocation with exponential backoff retries before triggering provider fallback.
    """
    try:
        return llm.invoke(messages)
    except Exception as e:
        logger.warning(f"LLM invocation encountered rate limit / API issue ({e}). Retrying...")
        raise e