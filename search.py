"""Web and academic search integrations (DuckDuckGo, Semantic Scholar, arXiv, OpenAlex)."""
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional

from config import OPENALEX_MAILTO

logger = logging.getLogger(__name__)

# ── Optional dependency flags ─────────────────────────────────────────────────
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False

try:
    from semanticscholar import SemanticScholar
    HAS_SEMANTIC_SCHOLAR = True
except ImportError:
    HAS_SEMANTIC_SCHOLAR = False

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False


# ── Web search ────────────────────────────────────────────────────────────────

def run_web_search(query: str) -> str:
    """Run a DuckDuckGo web search. Returns formatted results or "" on failure."""
    if not HAS_WEB_SEARCH:
        return "Web search unavailable. Install: pip install duckduckgo-search"
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        results = wrapper.results(query, max_results=5)
        if not results:
            return "No results found."
        formatted = "### Web Search Results\n\n"
        sources = []
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            snippet = res.get('snippet', 'No content available.')
            link = res.get('link', '#')
            formatted += f"**{i+1}. {title}**\n{snippet}\n\n"
            sources.append(f"{i+1}. [{title}]({link})")
        formatted += "---\n**Sources:**\n" + "\n".join(sources)
        return formatted
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return ""


# ── Academic search ───────────────────────────────────────────────────────────

def extract_academic_query(user_prompt: str, rag_context: str, llm) -> str:
    """Use the LLM to extract a tight academic search query from context."""
    prompt = f"""Extract an academic search query.

USER QUESTION:
{user_prompt}

CONTEXT:
{rag_context[:3000]}

TASK:
Write ONE short search query (5–7 keywords) for academic papers.

RULES:
- Use technical terms
- No explanations
- No punctuation
- Max 200 characters

OUTPUT:"""
    try:
        result = str(llm.invoke(prompt)).strip().split('\n')[0].strip('"\'')
        return result[:200] if len(result) > 200 else result
    except Exception as e:
        logger.warning(f"Failed to extract academic query: {e}")
        return user_prompt[:100]


def run_academic_search(query: str, max_results: int = 100,
                        rag_context: Optional[str] = None, llm=None) -> str:
    """Search Semantic Scholar, arXiv, and OpenAlex; rank and deduplicate results."""
    if rag_context and llm:
        query = extract_academic_query(query, rag_context, llm)

    all_papers = []

    # 1. Semantic Scholar
    if HAS_SEMANTIC_SCHOLAR:
        try:
            sch = SemanticScholar(timeout=5, retry=False)
            results = sch.search_paper(
                query,
                limit=min(max_results, 20),
                fields=['title', 'abstract', 'year', 'authors', 'citationCount', 'venue', 'url', 'openAccessPdf'],
            )
            for paper in results:
                if not paper.title:
                    continue
                authors = ", ".join([a.name for a in (paper.authors or [])[:3]])
                if len(paper.authors or []) > 3:
                    authors += " et al."
                abstract = paper.abstract or 'No abstract'
                if len(abstract) > 300:
                    abstract = abstract[:300] + '...'
                all_papers.append({
                    'source': 'Semantic Scholar',
                    'title': paper.title,
                    'authors': authors,
                    'year': paper.year or 'N/A',
                    'citations': paper.citationCount or 0,
                    'venue': paper.venue or '',
                    'abstract': abstract,
                    'url': paper.url or '',
                    'open_access': paper.openAccessPdf.get('url') if paper.openAccessPdf else None,
                })
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                logger.warning("Semantic Scholar rate limited")
            else:
                logger.warning(f"Semantic Scholar search failed: {e}")

    # 2. arXiv
    if HAS_ARXIV:
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=min(max_results, 100),
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for result in client.results(search):
                authors = ", ".join([a.name for a in result.authors[:3]])
                if len(result.authors) > 3:
                    authors += " et al."
                summary = result.summary
                all_papers.append({
                    'source': 'arXiv',
                    'title': result.title,
                    'authors': authors,
                    'year': result.published.year if result.published else 'N/A',
                    'citations': -1,
                    'venue': 'arXiv preprint',
                    'abstract': summary[:300] + '...' if len(summary) > 300 else summary,
                    'url': result.entry_id,
                    'open_access': result.pdf_url,
                })
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")

    # 3. OpenAlex
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://api.openalex.org/works?search={encoded}"
            f"&per_page={min(max_results, 100)}&sort=relevance_score:desc"
        )
        user_agent = 'OmnisciencePro/1.0'
        if OPENALEX_MAILTO:
            user_agent += f' (mailto:{OPENALEX_MAILTO})'
        req = urllib.request.Request(url, headers={'User-Agent': user_agent})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for work in data.get('results', []):
                authors_list = work.get('authorships', [])[:3]
                authors = ", ".join(
                    [a.get('author', {}).get('display_name', '') for a in authors_list]
                )
                if len(work.get('authorships', [])) > 3:
                    authors += " et al."

                abstract = ''
                inv_idx = work.get('abstract_inverted_index') or {}
                valid_idx = {w: p for w, p in inv_idx.items() if p}
                if valid_idx:
                    words = [''] * (max(max(p) for p in valid_idx.values()) + 1)
                    for word, positions in valid_idx.items():
                        for pos in positions:
                            words[pos] = word
                    joined = ' '.join(words)
                    abstract = (joined[:300] + '...') if len(joined) > 300 else joined

                all_papers.append({
                    'source': 'OpenAlex',
                    'title': work.get('title', 'Untitled'),
                    'authors': authors,
                    'year': work.get('publication_year', 'N/A'),
                    'citations': work.get('cited_by_count', 0),
                    'venue': (
                        work.get('primary_location', {})
                            .get('source', {})
                            .get('display_name', '')
                        if work.get('primary_location') else ''
                    ),
                    'abstract': abstract or 'No abstract available',
                    'url': work.get('doi') or work.get('id', ''),
                    'open_access': work.get('open_access', {}).get('oa_url'),
                })
    except Exception as e:
        logger.warning(f"OpenAlex search failed: {e}")

    if not all_papers:
        return "No academic papers found for this query."

    # Rank: relevance position + citation boost + recency
    current_year = datetime.now().year
    for i, paper in enumerate(all_papers):
        relevance = 100 - i
        citation = min(paper['citations'], 500) / 10 if paper['citations'] >= 0 else 0
        recency = (
            max(0, 10 - (current_year - paper['year']))
            if isinstance(paper['year'], int) else 0
        )
        paper['score'] = relevance + citation + recency

    seen, unique = set(), []
    for paper in sorted(all_papers, key=lambda x: x['score'], reverse=True):
        key = paper['title'].lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(paper)

    formatted = "### Academic Research Results\n\n"
    for i, paper in enumerate(unique[:30]):
        citations_str = f"Citations: {paper['citations']}" if paper['citations'] >= 0 else "Citations: N/A"
        formatted += f"**{i+1}. {paper['title']}**\n"
        formatted += f"   *{paper['authors']}* ({paper['year']}) | {citations_str} | {paper['source']}\n"
        if paper['venue']:
            formatted += f"   Venue: {paper['venue']}\n"
        formatted += f"   {paper['abstract']}\n"
        if paper['open_access']:
            formatted += f"   [Open Access PDF]({paper['open_access']})\n"
        elif paper['url']:
            formatted += f"   [Link]({paper['url']})\n"
        formatted += "\n"

    formatted += f"---\n*Found {len(unique)} unique papers from Semantic Scholar, arXiv, and OpenAlex*"
    return formatted
