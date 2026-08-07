import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"  # free-tier model on Groq
FAST_MODEL_NAME = "llama-3.1-8b-instant"


def get_llm(temperature: float = 0.0) -> ChatGroq: # we create a ChatGroq object with this
    """Single place that constructs the Groq chat model — so every node
    in the graph configures it the same way, and swapping models later
    means changing one line, not hunting through the codebase."""
    return ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def get_fast_llm(temperature: float = 0.0) -> ChatGroq:
    """Small model for router_node only — proven correct at 8B for simple 
    classification. evaluate_node and calculator expression-extraction were
    tried on this model and both failed harder reasoning cases — both use
    get_llm() instead."""
    return ChatGroq(model=FAST_MODEL_NAME, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))



@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
)
def invoke_with_retry(llm, messages):
    """Wraps an LLM call so a free-tier 429 doesn't crash the whole run.
    Retries only on RateLimitError specifically (not on every exception —
    a bad prompt or auth error should still fail immediately, not retry
    pointlessly). Exponential backoff: waits 2s, then 4s, then 8s, capped
    at 20s, giving Groq's per-minute quota time to reset."""
    return llm.invoke(messages)


if __name__ == "__main__":
    llm = get_llm()
    response = invoke_with_retry(llm, "In one sentence, what is a Transformer model?")
    print(response.content)