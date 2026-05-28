from ddgs import DDGS


def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
        for r in results
    ]
