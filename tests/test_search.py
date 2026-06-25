import json

import search


class _FakeResponse:
    """Minimal context-manager stand-in for urllib's urlopen() result."""
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _openalex_payload(abstract_index):
    return {
        "results": [{
            "title": "A Short Paper",
            "authorships": [{"author": {"display_name": "Jane Doe"}}],
            "publication_year": 2020,
            "cited_by_count": 3,
            "primary_location": {"source": {"display_name": "Test Journal"}},
            "abstract_inverted_index": abstract_index,
            "doi": "10.1234/example",
            "open_access": {"oa_url": None},
        }]
    }


def test_short_openalex_abstract_has_no_ellipsis(monkeypatch):
    """A short abstract must not get a misleading '...' truncation marker."""
    # Force OpenAlex to be the only contributing source.
    monkeypatch.setattr(search, "HAS_SEMANTIC_SCHOLAR", False)
    monkeypatch.setattr(search, "HAS_ARXIV", False)

    payload = _openalex_payload({"Hello": [0], "world": [1]})
    monkeypatch.setattr(
        search.urllib.request, "urlopen",
        lambda req, timeout=10: _FakeResponse(payload),
    )

    result = search.run_academic_search("anything")

    assert "Hello world" in result
    assert "Hello world..." not in result


def test_long_openalex_abstract_is_truncated(monkeypatch):
    """A long abstract is still truncated with '...' at 300 chars."""
    monkeypatch.setattr(search, "HAS_SEMANTIC_SCHOLAR", False)
    monkeypatch.setattr(search, "HAS_ARXIV", False)

    # 400 single-character "words" -> joined length well over 300.
    index = {f"w{i}": [i] for i in range(400)}
    payload = _openalex_payload(index)
    monkeypatch.setattr(
        search.urllib.request, "urlopen",
        lambda req, timeout=10: _FakeResponse(payload),
    )

    result = search.run_academic_search("anything")
    assert "..." in result
