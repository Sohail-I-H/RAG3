"""
search.py

Handles external academic paper discovery using:
1. arXiv API
2. DuckDuckGo Search
"""

import time
from typing import Dict, List

import arxiv
from duckduckgo_search import DDGS


# =============================================================================
# Search arXiv
# =============================================================================

def search_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search arXiv for research papers.
    """

    papers = []

    try:
        client = arxiv.Client(
            page_size=10,
            delay_seconds=3,
            num_retries=3
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for paper in client.results(search):

            papers.append({
                "title": paper.title,
                "summary": paper.summary.replace("\n", " "),
                "authors": ", ".join(
                    [author.name for author in paper.authors]
                ),
                "published": str(paper.published.date()),
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id
            })

            # polite delay
            time.sleep(0.5)

    except Exception as e:

        print("arXiv Error:", e)

    return papers


# =============================================================================
# DuckDuckGo Search
# =============================================================================

def search_web_papers(query: str,
                      max_results: int = 5) -> List[Dict]:
    """
    Search academic papers through DuckDuckGo.
    """

    results = []

    search_query = (
        f"{query} "
        "site:arxiv.org OR "
        "site:openreview.net OR "
        "site:ieeexplore.ieee.org OR "
        "site:acm.org"
    )

    try:

        with DDGS() as ddgs:

            response = ddgs.text(
                search_query,
                max_results=max_results
            )

            for item in response:

                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("href", ""),
                    "snippet": item.get("body", "")
                })

    except Exception as e:

        print("DuckDuckGo Error:", e)

    return results


# =============================================================================
# Combined Search
# =============================================================================

def discover_papers(topic: str) -> Dict:
    """
    Returns both arXiv and web results.
    """

    return {
        "arxiv": search_arxiv_papers(topic),
        "web": search_web_papers(topic)
    }


# =============================================================================
# Markdown Formatter
# =============================================================================

def format_results(results: Dict) -> str:
    """
    Convert search results into Markdown.
    """

    output = "# Related Research Papers\n\n"

    # -------------------------------------------------------------------------
    # arXiv
    # -------------------------------------------------------------------------

    output += "## arXiv Papers\n\n"

    if len(results["arxiv"]) == 0:

        output += (
            "_No papers found or arXiv API unavailable._\n\n"
        )

    else:

        for paper in results["arxiv"]:

            output += (
                f"### {paper['title']}\n\n"
                f"**Authors:** {paper['authors']}\n\n"
                f"**Published:** {paper['published']}\n\n"
                f"**PDF:** {paper['pdf_url']}\n\n"
                f"{paper['summary']}\n\n"
                "---\n\n"
            )

    # -------------------------------------------------------------------------
    # Web Search
    # -------------------------------------------------------------------------

    output += "\n## Web Results\n\n"

    if len(results["web"]) == 0:

        output += "_No web results found._"

    else:

        for item in results["web"]:

            output += (
                f"### {item['title']}\n\n"
                f"{item['snippet']}\n\n"
                f"{item['link']}\n\n"
                "---\n\n"
            )

    return output
