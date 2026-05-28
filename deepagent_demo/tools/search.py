from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo. Returns titles, snippets, and URLs."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    formatted = []
    for r in results:
        formatted.append(f"[{r.get('title', '')}]({r.get('href', '')})\n{r.get('body', '')}")
    return "\n\n".join(formatted) if formatted else "No results found."
