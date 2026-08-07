import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search_tool(query: str, max_results: int = 3) -> str:
    """Searches the live web for anything outside the local ArXiv corpus —
    recent papers, current benchmarks, follow-up work."""
    try:
        response = _client.search(query=query, max_results=max_results)
    except Exception as e:
        return f"Web search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No web results found."

    formatted = []
    for r in results:
        formatted.append(f"[{r['title']}]({r['url']})\n{r['content']}")
    return "\n\n---\n\n".join(formatted)

    # The general syntax is: "separator".join(iterable), where the iterable is usually a list of strings.
    # "Take every string in this list and glue them together, inserting the separator between consecutive strings."


if __name__ == "__main__":
    print(web_search_tool("recent papers building on LoRA fine-tuning 2026"))