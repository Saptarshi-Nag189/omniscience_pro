"""
Root conftest: stub out all heavy dependencies (Streamlit, LangChain, etc.)
before omniscience_pro.py is imported.  The functions under test only use
stdlib — this lets the test suite run in a lean environment without the full
dependency stack installed.
"""
import sys
import types
from unittest.mock import MagicMock


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _stub_if_missing(name: str, **attrs):
    if name not in sys.modules:
        sys.modules[name] = _make_module(name, **attrs)


# ── Streamlit ────────────────────────────────────────────────────────────────
_st = MagicMock(name="streamlit")
_st.cache_resource = lambda fn=None, **kw: (fn if fn else lambda f: f)
_stub_if_missing("streamlit", **{k: getattr(_st, k) for k in dir(_st)})

_comp = MagicMock(name="streamlit.components.v1")
_stub_if_missing("streamlit.components", v1=_comp)
_stub_if_missing("streamlit.components.v1")

# ── LangChain ────────────────────────────────────────────────────────────────
for _pkg in [
    "langchain",
    "langchain_core",
    "langchain_core.callbacks",
    "langchain_core.callbacks.base",
    "langchain_core.documents",
    "langchain_core.prompts",
    "langchain_chroma",
    "langchain_huggingface",
    "langchain_ollama",
    "langchain_text_splitters",
    "langchain_classic",
    "langchain_classic.chains",
    "langchain_classic.chains.retrieval_qa",
    "langchain_classic.chains.retrieval_qa.base",
    "langchain_community",
    "langchain_community.tools",
    "langchain_community.utilities",
]:
    _stub_if_missing(_pkg)

# Populate the classes/functions actually referenced at import time
sys.modules["langchain_core.callbacks.base"].BaseCallbackHandler = MagicMock
sys.modules["langchain_core.documents"].Document = MagicMock
sys.modules["langchain_core.prompts"].PromptTemplate = MagicMock
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock
sys.modules["langchain_text_splitters"].Language = MagicMock()
sys.modules["langchain_ollama"].OllamaLLM = MagicMock
sys.modules["langchain_huggingface"].HuggingFaceEmbeddings = MagicMock
sys.modules["langchain_chroma"].Chroma = MagicMock
sys.modules["langchain_classic.chains.retrieval_qa.base"].RetrievalQA = MagicMock
sys.modules["langchain_community.tools"].DuckDuckGoSearchRun = MagicMock
sys.modules["langchain_community.utilities"].DuckDuckGoSearchAPIWrapper = MagicMock

# ── Other heavy deps ──────────────────────────────────────────────────────────
for _pkg in ["chromadb", "chromadb.config", "pypdf", "pandas", "matplotlib",
             "matplotlib.pyplot", "PIL", "PIL.Image", "requests",
             "semanticscholar", "arxiv", "sentence_transformers"]:
    _stub_if_missing(_pkg)

sys.modules["chromadb.config"].Settings = MagicMock
