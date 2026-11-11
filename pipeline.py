"""
pipeline.py — Shared module for the privacy-policy assessment pipeline.

This module consolidates the functions developed in notebooks 01, 02 and 03,
thereby avoiding code duplication in the case-study notebook.

Stages:
  1. Ingestion and cleaning → load_policy(), save_policy()
  2. Segmentation           → chunk_policy(), save_chunks(), load_chunks()
  3. RAG assessment         → build_retriever(), run_audit(), save_audit()
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import io
import json
import math
import os
import re
import sys
import time
import uuid

# Enforce UTF-8 for standard output and error streams to prevent Windows encoding failures.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import chardet
import jsonschema
import requests
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import trafilatura
import pdfplumber
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_experimental.text_splitter import SemanticChunker
    _SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    _SEMANTIC_CHUNKER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# LLM provider pool with automatic rotation and expenditure limits.
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    """Identify HTTP 429 errors returned by Groq, Gemini or compatible providers."""
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "rate limit", "rate_limit", "quota exceeded", "resource exhausted"))


def _requires_manual_structured_output(provider: "LLMProvider") -> bool:
    """Return whether the provider requires manual JSON generation and validation."""
    if provider.manual_structured_output:
        return True
    if provider.name.lower().startswith("ollama"):
        return True
    try:
        from langchain_ollama import ChatOllama
        return isinstance(provider.llm, ChatOllama)
    except ImportError:
        return False


def _build_json_template(schema_dict: dict) -> dict:
    """Construct a template instance from a JSON Schema to guide the model."""
    import json as _json
    props = schema_dict.get("properties", {})
    defs  = schema_dict.get("$defs", {})
    template: dict = {}
    for name, prop in props.items():
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            prop = defs.get(ref_name, prop)
        if "enum" in prop:
            template[name] = " | ".join(str(v) for v in prop["enum"])
        elif prop.get("type") == "array":
            template[name] = []
        elif prop.get("type") == "number":
            template[name] = 0.0
        elif prop.get("type") == "integer":
            template[name] = 0
        elif prop.get("type") == "boolean":
            template[name] = False
        else:
            template[name] = f"<{name}>"
    return template


def _extract_json_object(text: str) -> str:
    """Extract the first valid, brace-balanced JSON object from a response that may contain reasoning, Markdown fences or other free text."""
    import json as _j
    text = text.strip()
    # Remove <think>...</think> blocks emitted by reasoning models such as Qwen or DeepSeek.
    text = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
    # Remove Markdown JSON fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{"):
        return text
    # Examine each opening brace and extract the balanced JSON object that follows it.
    for m in re.finditer(r"\{", text):
        candidate = text[m.start():]
        depth, in_str, escape, end = 0, False, False, -1
        for i, ch in enumerate(candidate):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            continue
        candidate = candidate[:end + 1]
        try:
            _j.loads(candidate)
            return candidate
        except _j.JSONDecodeError:
            continue
    return text


def _invoke_structured_for_provider(
    provider: "LLMProvider",
    schema_class: Any,
    messages: List,
    retrieved_chunks: Optional[List] = None,
    criterion: Optional[Dict] = None,
) -> Any:
    """Invoke structured generation using the provider-appropriate strategy, including the two-call protocol required by manually validated remote providers."""
    if not _requires_manual_structured_output(provider):
        return provider.llm.with_structured_output(schema_class).invoke(messages)

    import json as _json

    # Obtain the permitted verdict values from the schema.
    schema_dict = schema_class.model_json_schema()
    verdict_enum: List[str] = ["YES", "PARTIAL", "NOT_STATED", "NOT_APPLICABLE"]
    for _name, _prop in schema_dict.get("properties", {}).items():
        if _name == "verdict" and "enum" in _prop:
            verdict_enum = [str(v) for v in _prop["enum"]]
            break
    verdict_options = " | ".join(verdict_enum)

    # With a short response per call, the prompt may contain up to 7,500 characters.
    # The criterion remains at the end of the user prompt and is never truncated.
    MAX_PROMPT_CHARS = 7500

    def _trim(msgs: List) -> List:
        t = list(msgs)
        if len(t) < 2:
            return t
        sys_len = len(str(t[0][1]))
        budget = MAX_PROMPT_CHARS - sys_len - 300  # Reserve 300 characters for the final instruction.
        human = str(t[1][1])
        if len(human) <= budget:
            return t
        sep = "=" * 60
        parts = human.split(sep)
        if len(parts) >= 3:
            criterion_section = sep + sep.join(parts[2:])
            passages_budget = budget - len(criterion_section)
            passages = parts[1][:max(0, passages_budget)] + "\n[... additional passages omitted ...]"
            human = parts[0] + sep + passages + criterion_section
        else:
            tail = human[-2000:]
            head = human[:max(0, budget - 2000)]
            human = head + "\n[... passages omitted ...]\n" + tail
        return [t[0], ("human", human)]

    trimmed = _trim(messages)
    chunk_map = {c.chunk_id: c for c in (retrieved_chunks or [])}

    # ── Call 1: verdict, rationale and referenced chunk identifiers ─────────────────
    instr_1 = (
        f"Return ONLY this JSON object, without additional text or Markdown:\n"
        f'{{"verdict": "{verdict_options}", '
        f'"reasoning": "<brief, formal British English justification, max. 25 words>", '
        f'"chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"]}}\n'
        f"Write reasoning in formal British English. "
        f"In chunk_ids, provide the exact chunk_id values of the passages used to determine "
        f"the verdict (maximum 3; use an empty array [] if there is no relevant evidence)."
    )
    msgs_1 = trimmed + [("human", instr_1)]

    verdict: str | None = None
    reasoning: str = "No rationale was available."
    raw_chunk_ids: List[str] = []
    last_err_1: Exception | None = None

    for _attempt in range(3):
        raw = provider.llm.invoke(msgs_1)
        content = _extract_json_object(raw.content)
        try:
            data = _json.loads(content)
            v = data.get("verdict", "")
            if v in verdict_enum:
                verdict = v
                r = data.get("reasoning", "")
                if r and isinstance(r, str):
                    reasoning = r.strip()
                ids = data.get("chunk_ids", [])
                raw_chunk_ids = [str(i) for i in ids if isinstance(i, str)][:3]
                break
            last_err_1 = ValueError(f"Unrecognised verdict: {v!r}")
        except Exception as e:
            last_err_1 = e

    if verdict is None:
        raise ValueError(f"Call 1 (verdict) produced no valid response after three attempts: {last_err_1}")

    # Insert a brief pause between calls to avoid overloading the server.
    time.sleep(2)

    # ── Call 2: explanation based on the referenced chunks ──────────────────
    # Execute only when the model returns valid chunk identifiers from the retrieved set.
    # The second prompt contains only the cited chunks rather than the complete set.
    evidence: List[Dict] = []
    valid_ids = [cid for cid in raw_chunk_ids if cid in chunk_map]

    # For NOT_APPLICABLE and NOT_STATED, use the three most relevant retrieved chunks when
    # the model provides no chunk identifiers. Following U-shaped reordering, ranks map as follows:
    #   rank 1 → position 0; rank 2 → position -1; rank 3 → position 1
    if not valid_ids and verdict in ("NOT_APPLICABLE", "NOT_STATED") and retrieved_chunks:
        n = len(retrieved_chunks)
        top_relevant = [retrieved_chunks[0]]
        if n >= 2:
            top_relevant.append(retrieved_chunks[-1])
        if n >= 3:
            top_relevant.append(retrieved_chunks[1])
        valid_ids = [c.chunk_id for c in top_relevant if c.chunk_id in chunk_map]

    if valid_ids:
        small_context = "\n\n".join(
            f"[chunk_id={cid}]\n{chunk_map[cid].text}"
            for cid in valid_ids
        )
        crit_info = ""
        if criterion:
            crit_info = (
                f"Criterion {criterion['id']}: {criterion['name']}\n"
                f"Description: {criterion.get('description', '').strip()}\n"
                f"Minimum evidence required: {criterion['min_evidence']}\n"
                f"Question: {criterion['audit_question']}\n\n"
            )
        if verdict == "NOT_APPLICABLE":
            quote_instruction = (
                "copy the VERBATIM sentence in which the document explicitly states "
                "that this practice is not undertaken"
            )
        elif verdict == "NOT_STATED":
            quote_instruction = (
                "copy the passage most closely related to the criterion, even if it is "
                "insufficient to establish that the criterion is satisfied. If no passage "
                "is relevant, write: 'No relevant passages found.'"
            )
        else:
            quote_instruction = (
                f"copy the VERBATIM sentence or passage from the document that supports "
                f"the {verdict} verdict"
            )
        instr_2 = (
            f"{crit_info}"
            f"Verdict: {verdict}\n\n"
            f"Passages used to determine the verdict:\n{small_context}\n\n"
            f"Return ONLY this JSON object, without additional text or Markdown:\n"
            f'{{"quote": "<{quote_instruction}; do not paraphrase or translate>"}}'
        )
        msgs_2 = [messages[0], ("human", instr_2)]

        for _attempt in range(3):
            raw = provider.llm.invoke(msgs_2)
            content = _extract_json_object(raw.content)
            try:
                data = _json.loads(content)
                q = data.get("quote", "")
                if q and isinstance(q, str) and len(q) >= 10:
                    evidence = [{"chunk_id": cid, "quote": q.strip()} for cid in valid_ids]
                    break
            except Exception:
                pass

    # ── Assemble the final result ─────────────────────────────────────────────
    return schema_class.model_validate({
        "verdict": verdict,
        "reasoning": reasoning,
        "evidence_json": evidence,
    })


MAX_429_RETRIES    = 3   # Retries on the active provider before rotating to the next one.
RETRY_WAIT_SECONDS = 65  # Groq's TPM window is one minute; 65 seconds ensures that it resets.


@dataclass
class LLMProvider:
    name: str
    llm: Any
    max_calls: int
    # Some OpenAI-compatible endpoints do not implement the ``parsed`` and
    # ``refusal`` response fields expected by LangChain Structured Outputs.
    manual_structured_output: bool = False
    calls_made: int = 0
    _rate_limit_retries: int = field(default=0, repr=False)

    @property
    def exhausted(self) -> bool:
        return self.calls_made >= self.max_calls


class LLMPool:
    """Manage an ordered pool of LLM providers, rotating providers after rate-limit exhaustion or configured call limits."""

    def __init__(self, providers: List[LLMProvider], call_delay: float = 0.0) -> None:
        if not providers:
            raise ValueError("LLMPool requires at least one provider")
        self.providers = providers
        self._idx = 0
        self.call_delay = call_delay  # Delay between calls in seconds (0 disables the delay).
        self.model_usage_log: List[str] = []

    @property
    def current(self) -> LLMProvider:
        if self._idx >= len(self.providers):
            raise RuntimeError(
                "All LLM providers have reached their daily limits. "
                "Consult data/batch_log.csv to identify completed policies. "
                "Run the procedure again when the provider limits reset; processing will resume automatically."
            )
        return self.providers[self._idx]

    def _advance(self) -> None:
        self._idx += 1
        if self._idx >= len(self.providers):
            raise RuntimeError(
                "All LLM providers have reached their daily limits. "
                "Consult data/batch_log.csv to identify completed policies. "
                "Run the procedure again when the provider limits reset; processing will resume automatically."
            )
        print(f"\n  [LLMPool] Switching to provider: {self.current.name}", flush=True)

    def _handle_rate_limit(self, p: "LLMProvider") -> None:
        """Wait and retry the active provider before rotating to the next available provider."""
        p._rate_limit_retries += 1
        if p._rate_limit_retries <= MAX_429_RETRIES:
            print(
                f"\n  [LLMPool] Rate limit on {p.name} "
                f"(retry {p._rate_limit_retries}/{MAX_429_RETRIES}), "
                f"waiting {RETRY_WAIT_SECONDS}s...",
                flush=True,
            )
            time.sleep(RETRY_WAIT_SECONDS)
        else:
            print(
                f"\n  [LLMPool] Persistent rate limit on {p.name} after "
                f"{MAX_429_RETRIES} retries; switching provider...",
                flush=True,
            )
            p.calls_made = p.max_calls
            p._rate_limit_retries = 0
            self._advance()

    def invoke(self, messages: List) -> Any:
        """Invoke the active provider for unstructured generation."""
        while True:
            p = self.current
            if p.exhausted:
                self._advance()
                continue
            try:
                result = p.llm.invoke(messages)
                p.calls_made += 1
                self.model_usage_log.append(p.name)
                p._rate_limit_retries = 0
                if self.call_delay > 0:
                    time.sleep(self.call_delay)
                return result
            except Exception as exc:
                if _is_rate_limit(exc):
                    self._handle_rate_limit(p)
                else:
                    raise

    def invoke_structured(self, schema_class: Any, messages: List, retrieved_chunks: Optional[List] = None, criterion: Optional[Dict] = None) -> Any:
        """Invoke the active provider for structured criterion assessment."""
        while True:
            p = self.current
            if p.exhausted:
                self._advance()
                continue
            try:
                result = _invoke_structured_for_provider(p, schema_class, messages, retrieved_chunks, criterion)
                p.calls_made += 1
                self.model_usage_log.append(p.name)
                p._rate_limit_retries = 0
                if self.call_delay > 0:
                    time.sleep(self.call_delay)
                return result
            except Exception as exc:
                if _is_rate_limit(exc):
                    self._handle_rate_limit(p)
                else:
                    raise


def _structured_invoke(llm: Any, schema_class: Any, messages: List, retrieved_chunks: Optional[List] = None, criterion: Optional[Dict] = None) -> Any:
    """Provide a common structured-invocation interface for pooled and direct LangChain providers."""
    if isinstance(llm, LLMPool):
        return llm.invoke_structured(schema_class, messages, retrieved_chunks, criterion)
    return llm.with_structured_output(schema_class).invoke(messages)


def _direct_model_name(llm: Any) -> str:
    """Return the configured model identifier for a direct LangChain model."""
    for attribute in ("model_name", "model"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return llm.__class__.__name__


def _models_used_for_audit(llm: Any, usage_start: int) -> str:
    """Return the ordered, deduplicated model identifiers used by one audit."""
    if isinstance(llm, LLMPool):
        used = llm.model_usage_log[usage_start:]
    else:
        used = [_direct_model_name(llm)]
    unique = list(dict.fromkeys(used))
    return ", ".join(unique) if unique else "unknown"


def build_ollama_provider(model: str = "qwen2.5:7b", max_calls: int = 9999) -> "LLMProvider | None":
    """Construct a provider for a local Ollama deployment, returning ``None`` when the service is unavailable."""
    try:
        from langchain_ollama import ChatOllama
        import httpx
        # Confirm that Ollama is running.
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        resp.raise_for_status()
        llm = ChatOllama(model=model, temperature=0.0)
        return LLMProvider(name=f"ollama-{model}", llm=llm, max_calls=max_calls)
    except Exception as e:
        print(f"  [Pool] Ollama unavailable ({e})", flush=True)
        return None


def build_remote_provider(
    base_url: str,
    api_key: str,
    model: str = "gpt-oss",
    max_calls: int = 9999,
) -> "LLMProvider | None":
    """Construct a provider for a remote OpenAI-compatible service, returning ``None`` when the endpoint is unavailable."""
    try:
        from langchain_openai import ChatOpenAI
        import httpx
        httpx.get(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"},
                  timeout=5.0, verify=False)
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_tokens=2048,
            http_client=httpx.Client(verify=False, timeout=180.0),
        )
        return LLMProvider(
            name=f"remote-{model}",
            llm=llm,
            max_calls=max_calls,
            manual_structured_output=True,
        )
    except Exception as e:
        print(f"  [Pool] Remote server unavailable ({e})", flush=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Default paths, relative to the project directory.
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(".")
CRITERIA_DIR       = BASE_DIR / "criteria"
CHUNKS_DIR         = BASE_DIR / "data" / "chunks"
AUDITS_DIR         = BASE_DIR / "data" / "audits"
CLEAN_DIR          = BASE_DIR / "data" / "clean"
CHECKLIST_PATH     = CRITERIA_DIR / "checklist_v1.yaml"
OUTPUT_SCHEMA_PATH = CRITERIA_DIR / "output_schema.json"

# ─────────────────────────────────────────────────────────────────────────────
# Model configuration; values may be overridden by a calling notebook.
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDINGS_MODEL = "BAAI/bge-m3"          # High-quality multilingual model replacing all-MiniLM-L6-v2.
LLM_TEMPERATURE  = 0.0
TOP_K_CHUNKS     = 8    # Baseline used as a fallback when RETRIEVAL_MIN does not apply.
SUBCHUNK_SIZE    = 200  # Character length of sub-chunks indexed in FAISS (Small-to-Big).

# ── Enhanced retriever parameters ──────────────────────────────────────────
RRF_K                        = 60    # Standard k parameter for Reciprocal Rank Fusion.
RETRIEVAL_MIN                = 5     # Minimum number of chunks always returned.
RETRIEVAL_MAX                = 10    # Maximum chunks per query (Liu et al., 2024: more does not improve performance).
RETRIEVAL_THRESHOLD          = 0.20  # Minimum normalised RRF score for the standard pass.
RETRIEVAL_THRESHOLD_EXPANDED = 0.10  # threshold applied during the second verification pass
USE_HYDE                     = True  # HyDE: embed a hypothetical document rather than the question.


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 · INGESTION AND CLEANING
# ═════════════════════════════════════════════════════════════════════════════

_NOISE_TAGS = [
    "script", "style", "noscript", "svg",
    "nav", "header", "footer", "aside", "form",
]


@dataclass
class PolicyDoc:
    source: str
    source_type: str        # url_html | url_pdf | file_pdf | file_html
    fetched_at: str
    content_hash: str
    content_bytes: Optional[int]
    clean_text: str
    extra: Dict[str, Any]
    html_content: Optional[str] = field(default=None)  # decoded HTML retained only for web sources


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_url(url: str, timeout: int = 30) -> Tuple[bytes, Dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TFG-PrivacyAuditBot/1.0)"}
    session = _build_session()
    r = session.get(url, headers=headers, timeout=timeout, verify=True)
    r.raise_for_status()
    return r.content, {
        "content_type": r.headers.get("Content-Type", ""),
        "final_url": str(r.url),
    }


def is_probably_pdf(content_type: str, url: str, head_bytes: bytes) -> bool:
    ct = (content_type or "").lower()
    if "application/pdf" in ct:
        return True
    if url.lower().endswith(".pdf"):
        return True
    return head_bytes[:4] == b"%PDF"


def _detect_encoding(raw: bytes) -> str:
    result = chardet.detect(raw)
    enc = result.get("encoding") or "utf-8"
    confidence = result.get("confidence") or 0
    return enc if confidence >= 0.6 else "utf-8"


def _encoding_from_html_meta(html_bytes: bytes) -> Optional[str]:
    """Extract a declared character encoding from the first 2 KB of an HTML document."""
    head = html_bytes[:2048].decode("ascii", errors="ignore")
    m = re.search(r'charset=["\']?\s*([a-zA-Z0-9\-]+)', head, re.IGNORECASE)
    return m.group(1) if m else None


def _resolve_encoding(html_bytes: bytes, content_type: str = "") -> str:
    """Resolve HTML character encoding by prioritising the HTTP header, then HTML metadata, and finally statistical detection."""
    ct_lower = content_type.lower()
    if "charset=" in ct_lower:
        enc = ct_lower.split("charset=")[-1].strip().split(";")[0].strip()
        if enc:
            return enc
    meta_enc = _encoding_from_html_meta(html_bytes)
    if meta_enc:
        return meta_enc
    return _detect_encoding(html_bytes)


def extract_clean_text_from_html(
    html_bytes: bytes,
    url: Optional[str] = None,
    content_type: str = "",
) -> str:
    encoding = _resolve_encoding(html_bytes, content_type)
    html = html_bytes.decode(encoding, errors="replace")
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )
    if extracted and len(extracted.strip()) > 300:
        return normalize_whitespace(extracted)
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    return normalize_whitespace(soup.get_text("\n"))


def extract_html_blocks(html: str) -> List[Dict[str, Any]]:
    """Extract structural HTML blocks whilst preserving heading hierarchy, paragraphs and list items."""
    soup = BeautifulSoup(html, "html.parser")
    for noise in soup(_NOISE_TAGS):
        noise.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"id": re.compile(r"content|main|body", re.I)})
        or soup.find(attrs={"class": re.compile(r"content|main|body|policy|privacy", re.I)})
        or soup.body
        or soup
    )

    blocks: List[Dict[str, Any]] = []

    def _only_bold(node) -> bool:
        children = [c for c in node.children if getattr(c, "name", None)]
        return len(children) == 1 and children[0].name in ("strong", "b")

    def walk(el):
        for child in el.children:
            tag = getattr(child, "name", None)
            if tag is None:
                continue
            tag = tag.lower()

            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                text = child.get_text(" ", strip=True)
                if len(text) >= 4:
                    blocks.append({"tag": tag, "level": int(tag[1]), "text": text, "is_heading": True})

            elif tag == "p":
                text = child.get_text(" ", strip=True)
                if not text or len(text) < 10:
                    continue
                if _only_bold(child) and len(text) <= 200:
                    blocks.append({"tag": "strong", "level": 3, "text": text, "is_heading": True})
                elif text.startswith("¿") and text.endswith("?") and len(text) <= 150:
                    blocks.append({"tag": "p-question", "level": 3, "text": text, "is_heading": True})
                elif len(text) >= 25:
                    blocks.append({"tag": "p", "level": 0, "text": text, "is_heading": False})

            elif tag in ("ul", "ol"):
                for li in child.find_all("li", recursive=False):
                    text = li.get_text(" ", strip=True)
                    if len(text) >= 10:
                        blocks.append({"tag": "li", "level": 0, "text": "• " + text, "is_heading": False})

            elif tag == "table":
                # Convert each row into a text block.
                # Use header cells (th) as a "Field: Value" prefix for data rows.
                header_cells: List[str] = []
                for tr in child.find_all("tr"):
                    ths = [th.get_text(" ", strip=True) for th in tr.find_all("th")]
                    tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                    if ths:
                        header_cells = [h for h in ths if h]
                        row_text = " | ".join(header_cells)
                        if len(row_text) >= 4:
                            blocks.append({"tag": "th", "level": 0, "text": row_text, "is_heading": False})
                    elif tds:
                        cells = [c for c in tds if c]
                        if not cells:
                            continue
                        if header_cells and len(header_cells) == len(cells):
                            row_text = " | ".join(f"{h}: {v}" for h, v in zip(header_cells, cells))
                        else:
                            row_text = " | ".join(cells)
                        if len(row_text) >= 10:
                            blocks.append({"tag": "td", "level": 0, "text": row_text, "is_heading": False})

            elif tag in ("div", "section", "article", "main"):
                walk(child)

    walk(main)
    return blocks


def _group_blocks_by_heading(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group content blocks under their preceding headings without separating a heading from its associated content."""
    sections: List[Dict[str, Any]] = []
    current_heading = ""
    current_level = 0
    current_lines: List[str] = []
    sec_idx = 0

    def flush():
        nonlocal sec_idx
        if current_lines:
            sections.append({
                "heading": current_heading,
                "level": current_level,
                "section_index": sec_idx,
                "content": "\n".join(current_lines),
            })
            sec_idx += 1

    for block in blocks:
        if block["is_heading"]:
            flush()
            current_heading = block["text"]
            current_level = block["level"]
            current_lines = []
        else:
            current_lines.append(block["text"])

    flush()
    return sections


def _extract_page_text(page) -> str:
    """Extract flowing text and tabular content from a PDF page so that both remain available to the segmenter."""
    text = page.extract_text() or ""

    tables = page.extract_tables() or []
    table_lines = []
    for table in tables:
        for row in (table or []):
            cells = [str(cell).strip() for cell in (row or []) if cell and str(cell).strip()]
            if cells:
                table_lines.append(" | ".join(cells))

    if table_lines:
        text = text + "\n" + "\n".join(table_lines)

    return text


def extract_clean_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [_extract_page_text(p) for p in pdf.pages]
        text = normalize_whitespace("\n".join(pages))
        if len(text) > 300:
            return text
    except Exception:
        pass
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = normalize_whitespace("\n".join(pages))
        if len(text) > 300:
            return text
    except Exception:
        pass
    raise ValueError(
        "Text could not be extracted from the PDF, which may contain scanned images. "
        "Optical character recognition is required to process this document."
    )


def looks_like_privacy_policy(text: str) -> bool:
    t = text.lower()
    keywords = [
        "política de privacidad", "privacy policy", "rgpd", "gdpr",
        "responsable del tratamiento", "delegado de protección de datos",
        "derechos", "finalidad", "legitimación", "base jurídica",
        "conservación", "retención", "destinatarios", "transferencias",
        "cookies",
    ]
    hits = sum(1 for kw in keywords if kw in t)
    return hits >= 4 and len(text) > 800


def load_policy(source: str) -> PolicyDoc:
    fetched_at = now_iso()

    if is_url(source):
        content, info = fetch_url(source)
        final_url = info["final_url"]
        ct = info["content_type"]
        h = sha256_bytes(content)
        if is_probably_pdf(ct, final_url, content):
            clean_text = extract_clean_text_from_pdf_bytes(content)
            stype = "url_pdf"
            html_decoded = None
        else:
            clean_text = extract_clean_text_from_html(content, url=final_url, content_type=ct)
            stype = "url_html"
            enc = _resolve_encoding(content, ct)
            html_decoded = content.decode(enc, errors="replace")
        return PolicyDoc(
            source=final_url,
            source_type=stype,
            fetched_at=fetched_at,
            content_hash=h,
            content_bytes=len(content),
            clean_text=clean_text,
            extra={"content_type": ct},
            html_content=html_decoded,
        )

    if source.lower().endswith(".pdf") and os.path.exists(source):
        with open(source, "rb") as f:
            b = f.read()
        return PolicyDoc(
            source=os.path.abspath(source),
            source_type="file_pdf",
            fetched_at=fetched_at,
            content_hash=sha256_bytes(b),
            content_bytes=len(b),
            clean_text=extract_clean_text_from_pdf_bytes(b),
            extra={},
        )

    if source.lower().endswith((".html", ".htm")) and os.path.exists(source):
        with open(source, "rb") as f:
            b = f.read()
        enc = _resolve_encoding(b, "")
        html_decoded = b.decode(enc, errors="replace")
        return PolicyDoc(
            source=os.path.abspath(source),
            source_type="file_html",
            fetched_at=fetched_at,
            content_hash=sha256_bytes(b),
            content_bytes=len(b),
            clean_text=extract_clean_text_from_html(b, content_type=""),
            extra={},
            html_content=html_decoded,  # Required by chunk_policy_hybrid.
        )

    raise ValueError(
        "Unsupported source: provide an HTTP(S) URL or a path to a .pdf, .html or .htm file"
    )


def save_policy(doc: PolicyDoc, out_dir: str = "data/clean") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    base = doc.content_hash[:16]
    txt_path = os.path.join(out_dir, base + ".txt")
    meta_path = os.path.join(out_dir, base + ".json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(doc.clean_text)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": doc.source,
                "source_type": doc.source_type,
                "fetched_at": doc.fetched_at,
                "content_hash": doc.content_hash,
                "content_bytes": doc.content_bytes,
                "extra": doc.extra,
                "text_length": len(doc.clean_text),
                "looks_like_privacy_policy": looks_like_privacy_policy(doc.clean_text),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return txt_path, meta_path


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 · SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

_NUMBERED = re.compile(
    r"^(?:"
    r"(?:\d{1,2}[\.)\-]\s)"
    r"|(?:[IVXLC]+[\.)\-]\s)"
    r"|(?:[A-Z][\.)\-]\s)"
    r")",
    re.UNICODE,
)


@dataclass
class Chunk:
    """Represent a text passage with complete provenance to its source document."""
    chunk_id: str
    doc_hash: str
    doc_source: str
    chunk_index: int
    total_chunks: int
    section_title: str
    section_index: int
    char_start: int
    char_end: int
    text: str
    embed_text: Optional[str] = field(default=None)
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


def _build_embed_texts(chunks: List["Chunk"]) -> None:
    """Construct context-enriched embedding text whilst retaining clean passage text for lexical retrieval and LLM assessment."""
    for i, chunk in enumerate(chunks):
        parts: List[str] = []

        if chunk.section_title:
            parts.append(f"[Section: {chunk.section_title}]")

        prev = chunks[i - 1] if i > 0 and chunks[i - 1].section_index == chunk.section_index else None
        nxt  = chunks[i + 1] if i < len(chunks) - 1 and chunks[i + 1].section_index == chunk.section_index else None

        if prev:
            parts.append(f"[Previous: {prev.text[:200].strip()}]")

        parts.append(chunk.text)

        if nxt:
            parts.append(f"[Next: {nxt.text[:200].strip()}]")

        chunk.embed_text = "\n\n".join(parts)


def _is_section_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if line.endswith(".") and not _NUMBERED.match(line):
        return False
    if line.startswith(("–", "-", "•", "*")):
        return False
    if _NUMBERED.match(line):
        return True
    if line.startswith("¿") and line.endswith("?") and len(line) <= 120:
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 6:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio >= 0.60 and len(line) <= 80:
            return True
    return False


def split_into_sections(text: str) -> List[Dict[str, Any]]:
    lines = text.split("\n")
    sections: List[Dict[str, Any]] = []
    current_title = ""
    current_lines: List[str] = []
    current_start = 0
    offset = 0

    def flush(title, lines_buf, start, end):
        body = "\n".join(lines_buf).strip()
        if body:
            sections.append({"title": title, "text": body, "char_start": start, "char_end": end})

    for line in lines:
        line_len = len(line) + 1
        if _is_section_header(line):
            flush(current_title, current_lines, current_start, offset)
            current_title = line.strip()
            current_lines = []
            current_start = offset
        else:
            current_lines.append(line)
        offset += line_len

    flush(current_title, current_lines, current_start, offset)

    if not sections:
        sections = [{"title": "", "text": text, "char_start": 0, "char_end": len(text)}]

    return sections


def chunk_policy(
    doc: PolicyDoc,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    sections = split_into_sections(doc.clean_text)
    chunks: List[Chunk] = []
    chunk_index = 0

    for sec_idx, section in enumerate(sections):
        sec_text = section["text"]
        sec_start = section["char_start"]
        sub_texts = splitter.split_text(sec_text)
        search_from = 0

        for sub_text in sub_texts:
            pos = sec_text.find(sub_text.rstrip(), search_from)
            if pos == -1:
                pos = sec_text.find(sub_text[:40], search_from)
            if pos == -1:
                pos = search_from
            abs_start = sec_start + pos
            abs_end = abs_start + len(sub_text)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_hash=doc.content_hash,
                    doc_source=doc.source,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    section_title=section["title"],
                    section_index=sec_idx,
                    char_start=abs_start,
                    char_end=abs_end,
                    text=sub_text,
                )
            )
            search_from = max(pos + len(sub_text) - chunk_overlap, pos + 1)
            chunk_index += 1

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
    _build_embed_texts(chunks)
    return chunks


def chunk_policy_semantic(
    doc: PolicyDoc,
    breakpoint_type: str = "percentile",
    breakpoint_threshold: float = 95.0,
    min_chunk_chars: int = 100,
) -> List[Chunk]:
    """Segment a policy by combining document structure with semantic similarity, ensuring that passages do not cross section boundaries."""
    if not _SEMANTIC_CHUNKER_AVAILABLE:
        raise ImportError(
            "langchain-experimental is not installed. "
            "Run: pip install langchain-experimental"
        )

    embeddings = _get_embeddings()
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=breakpoint_threshold,
    )

    sections = split_into_sections(doc.clean_text)
    chunks: List[Chunk] = []
    chunk_index = 0

    for sec_idx, section in enumerate(sections):
        sec_text = section["text"]
        if not sec_text.strip():
            continue

        try:
            sub_docs = splitter.create_documents([sec_text])
        except Exception:
            # Fallback: represent a section that is too short for embeddings as a single chunk.
            sub_docs = [Document(page_content=sec_text)]

        search_from = 0
        for sub_doc in sub_docs:
            text = sub_doc.page_content.strip()
            if len(text) < min_chunk_chars:
                continue

            pos = sec_text.find(text[:60], search_from)
            if pos == -1:
                pos = search_from
            abs_start = section["char_start"] + pos
            abs_end = abs_start + len(text)

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_hash=doc.content_hash,
                doc_source=doc.source,
                chunk_index=chunk_index,
                total_chunks=0,
                section_title=section["title"],
                section_index=sec_idx,
                char_start=abs_start,
                char_end=abs_end,
                text=text,
            ))
            search_from = max(pos + len(text) - 60, pos + 1)
            chunk_index += 1

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
    _build_embed_texts(chunks)
    return chunks


def chunk_policy_hybrid(
    doc: PolicyDoc,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    """Apply two-stage structural and size-based segmentation whilst preserving section titles in every resulting passage."""
    if doc.source_type in ("url_pdf", "file_pdf"):
        return chunk_policy(doc, chunk_size, chunk_overlap)

    # ── Attempt structured extraction from HTML ────────────────────────────
    # Use the DOM when the HTML provides meaningful structure (h1–h6, paragraphs and lists).
    # For flat HTML based on <br>/|||, as in OPP-115, fall back to
    # split_into_sections over the cleaned text, which identifies sections
    # through numbering, capitalisation and question-pattern heuristics without discarding
    # the hybrid pipeline; enriched embed_text remains available.
    sections = None

    if doc.html_content:
        try:
            blocks = extract_html_blocks(doc.html_content)
            if len(blocks) >= 3:
                block_chars = sum(len(b.get("text", "")) for b in blocks)
                clean_chars = len(doc.clean_text) or 1
                # Only use DOM path if blocks cover ≥30% of document text.
                # Below that threshold the extractor found only minor elements
                # (e.g. <li> bullet lists) while the bulk of the policy text
                # lives in untagged nodes — falling back gives much better coverage.
                if block_chars / clean_chars >= 0.30:
                    sections = _group_blocks_by_heading(blocks) or None
        except Exception:
            pass

    # Structural fallback: apply heuristics to cleaned text rather than using an unstructured baseline.
    if not sections:
        raw_sections = split_into_sections(doc.clean_text)
        sections = [
            {
                "heading": s["title"],
                "content": s["text"],
                "section_index": idx,
            }
            for idx, s in enumerate(raw_sections)
            if s["text"].strip()
        ]

    if not sections:
        return chunk_policy(doc, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["|||", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Chunk] = []
    chunk_index = 0

    for sec in sections:
        heading = sec["heading"]
        content = sec["content"].strip()
        sec_idx = sec["section_index"]

        if not content:
            continue

        sub_texts = [content] if len(content) <= chunk_size else splitter.split_text(content)

        for sub_text in sub_texts:
            if not sub_text.strip():
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_hash=doc.content_hash,
                doc_source=doc.source,
                chunk_index=chunk_index,
                total_chunks=0,
                section_title=heading,
                section_index=sec_idx,
                char_start=0,
                char_end=len(sub_text),
                text=sub_text,
            ))
            chunk_index += 1

    if not chunks:
        return chunk_policy(doc, chunk_size, chunk_overlap)

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    _build_embed_texts(chunks)
    return chunks


def save_chunks(chunks: List[Chunk], out_dir: str = "data/chunks") -> str:
    if not chunks:
        raise ValueError("The chunk list is empty")
    os.makedirs(out_dir, exist_ok=True)
    base = chunks[0].doc_hash[:16]
    path = os.path.join(out_dir, base + "_chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    return path


def load_chunks(path: str) -> List[Chunk]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**{k: v for k, v in d.items() if k != "char_count"}) for d in data]


def chunk_stats(chunks: List[Chunk]) -> Dict[str, Any]:
    sizes = [c.char_count for c in chunks]
    sections = sorted({c.section_title for c in chunks})
    return {
        "total_chunks": len(chunks),
        "total_chars": sum(sizes),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "avg_chars": round(sum(sizes) / len(sizes), 1),
        "sections_detected": len(sections),
        "section_titles": sections,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 · RAG ASSESSMENT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert data-protection auditor specialising in the GDPR.
Your task is to assess whether a privacy policy satisfies a specific criterion.

OUTPUT LANGUAGE:
- Always write the reasoning field in formal, academic British English.
- Keep every quotation in the quote field in the document's original language and
  reproduce it verbatim, without translation or paraphrase.

STRICT RULES:
- Base your response EXCLUSIVELY on the document passages provided.
- Do NOT use external knowledge or assume anything that is not stated in the text.
- Quotations must be VERBATIM reproductions of the document's exact wording.
- If the evidence is insufficient, return NOT_STATED; never invent evidence.
- If the criterion clearly does not apply to the document, return NOT_APPLICABLE.
- Return PARTIAL when the topic is mentioned only vaguely, incompletely or ambiguously.

VERDICTS — READ CAREFULLY:
- YES: the evidence DIRECTLY addresses the criterion. A passing reference in a list of
  rights without any explanation is PARTIAL, not YES. The minimum standard is the GDPR
  requirement (Articles 13–14), rather than the standard adopted by large organisations.
- PARTIAL: the document addresses the topic incompletely, vaguely or without elaboration.
- NOT_STATED: the document contains no statement concerning the criterion; it is silent.
- NOT_APPLICABLE: the document EXPLICITLY states that the practice does not occur (for example,
  that no automated decision-making is undertaken or that data are not disclosed to third parties).
  IMPORTANT: NOT_APPLICABLE is not NOT_STATED. NOT_STATED denotes silence; NOT_APPLICABLE denotes an explicit denial.

EVIDENCE:
- Quote ONLY passages that specifically address the criterion's question.
- Disregard passages concerning a different topic entirely.
- YES: include quotations that establish that the criterion is satisfied.
- PARTIAL: include the partial or incomplete quotations identified.
- NOT_STATED: include the passages most closely related to the criterion, even if they are
  insufficient, so that a human reviewer can inspect them. Return an empty array [] only
  when no passage is even remotely relevant.
- NOT_APPLICABLE: include the verbatim quotation in which the document explicitly denies the practice.

EVIDENCE FORMAT:
Return an array of verbatim documentary quotations in evidence_json:
[{"chunk_id": "chunk-id", "quote": "verbatim quotation from the document"}, ...]
Include as many quotations as necessary. Return an empty array [] when there is no evidence."""


class CriterionLLMOutput(BaseModel):
    verdict: Literal["YES", "PARTIAL", "NOT_STATED", "NOT_APPLICABLE"] = Field(
        description=(
            "YES = clear and direct evidence. "
            "PARTIAL = ambiguous or incomplete evidence. "
            "NOT_STATED = no evidence in the document. "
            "NOT_APPLICABLE = the criterion does not apply to this document."
        )
    )
    reasoning: str = Field(
        description=(
            "Brief justification in formal British English, based ONLY on the "
            "provided passages (maximum 25 words)"
        ),
    )
    evidence_json: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            'Array of documentary evidence:'
            ' [{"chunk_id": "...", "quote": "verbatim text"}, ...].'
            ' Include as many quotations as necessary to support the verdict.'
            ' The quote field must reproduce the document verbatim, without paraphrase or translation.'
            ' NOT_APPLICABLE: include the quotation in which the document explicitly denies the practice.'
            ' NOT_STATED: include the passages most closely related to the criterion, even if they'
            ' are insufficient to establish that it is satisfied. Return [] only when no retrieved'
            ' passage is even remotely relevant.'
        )
    )


class ConsultaLLMOutput(BaseModel):
    verdict: Literal["YES", "PARTIAL", "NOT_STATED", "NOT_APPLICABLE"] = Field(
        description=(
            "YES = the document answers the question clearly. "
            "PARTIAL = the document answers the question partially or ambiguously. "
            "NOT_STATED = the information is absent from the document. "
            "NOT_APPLICABLE = the question does not apply because the organisation explicitly denies the practice."
        )
    )
    reasoning: str = Field(
        description=(
            "Brief justification in formal British English, based ONLY on the "
            "provided passages (maximum 25 words)"
        ),
    )
    evidence_json: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            'Verbatim documentary quotations: [{"chunk_id": "...", "quote": "verbatim text"}, ...].'
            ' Include quotations for every verdict.'
            ' NOT_APPLICABLE: quote the statement that explicitly denies the practice.'
            ' NOT_STATED: quote the closest passage found, even if it does not answer the question.'
            ' Return [] only when no passage is remotely relevant.'
        )
    )


QA_STRUCTURED_SYSTEM_PROMPT = """You are an expert assistant specialising in privacy-policy analysis.
Answer the question EXCLUSIVELY on the basis of the document passages provided.

OUTPUT LANGUAGE:
- Always write the reasoning field in formal, academic British English.
- Keep every quotation in the quote field in the document's original language and
  reproduce it verbatim, without translation or paraphrase.

VERDICT (select one):
- YES: the document answers the question clearly and completely.
- PARTIAL: the document answers the question partially or ambiguously.
- NOT_STATED: the information does not appear in any retrieved passage.
- NOT_APPLICABLE: the question does not apply because the document explicitly denies the practice.

EVIDENCE (required for every verdict):
- YES / PARTIAL: provide a verbatim quotation supporting the answer.
- NOT_STATED: quote the passage most closely related to the topic, even if it does not answer
  the question. Return [] if no passage is even minimally relevant.
- NOT_APPLICABLE: quote the statement that explicitly denies the practice.

RULES:
- Use ONLY the passages provided. Do not use external knowledge.
- Quotations must reproduce the document verbatim, without paraphrase or translation."""


def _parse_evidence_json(evidence_json_str: Any) -> List[Dict]:
    """Parse and validate documentary evidence returned by the LLM, discarding malformed or substantively empty items."""
    try:
        if isinstance(evidence_json_str, list):
            items = evidence_json_str
        else:
            raw = str(evidence_json_str).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            items = json.loads(raw)
        if not isinstance(items, list):
            return []
        return [
            {
                "chunk_id": e.get("chunk_id", ""),
                "quote": e.get("quote", ""),
                "relevance_score": None,
            }
            for e in items
            if isinstance(e, dict) and len(e.get("quote", "")) >= 10
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


HYDE_SYSTEM_PROMPT = """You specialise in drafting privacy policies.
Generate a passage of two or three sentences showing how a genuine, well-drafted privacy
policy would address the specified topic. Emulate the corporate style and vocabulary used
in privacy policies rather than the formal language of the GDPR. Return ONLY the passage,
without an introduction or explanation."""


def generate_hyde_query_qa(question: str, llm) -> str:
    """Generate a hypothetical policy passage for expert-query retrieval and fall back to the original question if generation fails."""
    prompt = (
        f"User question: {question}\n\n"
        f"Generate the privacy-policy passage (two or three sentences) that would answer this question."
    )
    try:
        response = llm.invoke([("system", HYDE_SYSTEM_PROMPT), ("human", prompt)])
        return response.content.strip()
    except Exception:
        return question  # Fall back to the original question.


def generate_hyde_query(criterion: Dict, llm) -> str:
    """Generate a criterion-specific hypothetical policy passage to improve semantic retrieval through Hypothetical Document Embeddings."""
    lang = criterion.get("_language", "es")
    lang_line = "Write the passage in English." if lang == "en" else "Write the passage in Spanish."
    prompt = (
        f"Topic to address: {criterion['name']}\n"
        f"Description: {criterion['description'].strip()}\n\n"
        f"Generate the privacy-policy passage. {lang_line}"
    )
    try:
        response = llm.invoke([("system", HYDE_SYSTEM_PROMPT), ("human", prompt)])
        return response.content.strip()
    except Exception:
        return criterion["audit_question"]  # fall back to the original query


def load_checklist(path: Path = CHECKLIST_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_output_schema(path: Path = OUTPUT_SCHEMA_PATH) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_retriever(chunks: List[Chunk], top_k: int = TOP_K_CHUNKS) -> Callable:
    """Construct a hybrid retriever combining Small-to-Big semantic retrieval, BM25 lexical retrieval and Reciprocal Rank Fusion."""
    embeddings = _get_embeddings()
    chunk_map = {c.chunk_id: c for c in chunks}

    # ── Small-to-Big retrieval: construct sub-chunks for FAISS ─────────────────────────
    # Divide each parent chunk at sentence boundaries, which provide natural semantic divisions.
    # Use a fixed SUBCHUNK_SIZE character window for exceptionally long sentences.
    # Prefix each sub-chunk with its parent section title to retain contextual information.
    _sent_re = re.compile(r'(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÜÑ\"\(])')

    def _split_sentences(text: str) -> List[str]:
        sub_texts: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sent in _sent_re.split(line):
                sent = sent.strip()
                if not sent or len(sent) < 20:
                    continue
                if len(sent) <= SUBCHUNK_SIZE:
                    sub_texts.append(sent)
                else:
                    for i in range(0, len(sent), SUBCHUNK_SIZE):
                        sub = sent[i:i + SUBCHUNK_SIZE].strip()
                        if sub and len(sub) >= 20:
                            sub_texts.append(sub)
        return sub_texts or [text[:SUBCHUNK_SIZE]]  # Fallback: always return at least one sub-chunk.

    sub_docs: List[Document] = []
    for c in chunks:
        section_prefix = f"[Section: {c.section_title}]\n" if c.section_title else ""
        for sub_text in _split_sentences(c.text):
            embed_text = section_prefix + sub_text
            sub_docs.append(Document(
                page_content=embed_text,
                metadata={"parent_chunk_id": c.chunk_id, "chunk_index": c.chunk_index},
            ))

    vector_store = FAISS.from_documents(sub_docs, embeddings)

    # ── Apply BM25 to parent chunks using cleaned text without prefixes ──────────────────
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        query: str,
        semantic_query: Optional[str] = None,
        expanded: bool = False,
        k: int = RETRIEVAL_MAX,
    ) -> List[Chunk]:
        threshold = RETRIEVAL_THRESHOLD_EXPANDED if expanded else RETRIEVAL_THRESHOLD
        faiss_query = semantic_query if semantic_query else query

        # FAISS searches sub-chunks; retain the minimum L2 distance for each parent chunk.
        # Retrieve up to k*5 sub-chunks to improve coverage of k distinct parent chunks.
        faiss_sub = vector_store.similarity_search_with_score(faiss_query, k=k * 5)
        best_l2: Dict[str, float] = {}
        for doc, dist in faiss_sub:
            pid = doc.metadata["parent_chunk_id"]
            if pid not in best_l2 or dist < best_l2[pid]:
                best_l2[pid] = dist

        # Order parent chunks by ascending L2 distance, where a lower value indicates greater relevance.
        faiss_ranked = [
            chunk_map[pid]
            for pid in sorted(best_l2, key=lambda p: best_l2[p])
            if pid in chunk_map
        ]

        # BM25: a higher score indicates greater relevance.
        tokens = query.lower().split()
        bm25_scores_arr = bm25.get_scores(tokens)
        bm25_ranked_idx = sorted(
            range(len(bm25_scores_arr)), key=lambda i: bm25_scores_arr[i], reverse=True
        )[:k]
        bm25_ranked = [chunks[i] for i in bm25_ranked_idx if bm25_scores_arr[i] > 0]

        # ── Apply RRF to parent chunks ────────────────────────────────────
        rrf: Dict[str, float] = {}
        for rank, chunk in enumerate(faiss_ranked):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, chunk in enumerate(bm25_ranked):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        if not rrf:
            return []

        max_rrf = max(rrf.values())
        normalized = {cid: s / max_rrf for cid, s in rrf.items()}
        sorted_ids = sorted(normalized, key=lambda cid: normalized[cid], reverse=True)

        above = [cid for cid in sorted_ids if normalized[cid] >= threshold]
        below = [cid for cid in sorted_ids if normalized[cid] < threshold]

        result_ids = above[:k]
        if len(result_ids) < RETRIEVAL_MIN:
            result_ids += below[:RETRIEVAL_MIN - len(result_ids)]

        return [chunk_map[cid] for cid in result_ids if cid in chunk_map]

    return retrieve


def _u_shape_reorder(chunks: List[Chunk]) -> List[Chunk]:
    """Redistribute ranked passages towards the beginning and end of the context to mitigate the lost-in-the-middle effect."""
    if len(chunks) <= 2:
        return chunks
    result: List[Optional[Chunk]] = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1
    return [c for c in result if c is not None]


def build_user_prompt(criterion: Dict, retrieved_chunks: List[Chunk]) -> str:
    chunks_text = "\n\n".join(
        f"[PASSAGE {i + 1} · chunk_id={c.chunk_id}]\n{c.text}"
        for i, c in enumerate(retrieved_chunks)
    )
    variants = criterion.get("query_variants", [])
    keywords_line = (
        f"Keywords and synonyms to consider: {' | '.join(variants)}\n"
        if variants else ""
    )
    # Repeat the criterion block before and after the retrieved passages.
    # Query-aware contextualisation (Liu et al., 2024): place the question at the beginning
    # and end of the context to mitigate primacy and recency effects and improve model attention
    # to passages located in the middle of the context.
    criterion_block = (
        f"CRITERION TO ASSESS ({criterion['id']}): {criterion['name']}\n"
        f"Description: {criterion.get('description', '').strip()}\n"
        f"Minimum evidence required: {criterion['min_evidence']}\n"
        f"{keywords_line}"
        f"QUESTION: {criterion['audit_question']}"
    )
    return (
        f"{criterion_block}\n\n"
        f"DOCUMENT PASSAGES:\n"
        f"{'=' * 60}\n"
        f"{chunks_text}\n"
        f"{'=' * 60}\n\n"
        f"{criterion_block}"
    )


def _multi_query_retrieve(
    criterion: Dict,
    retrieve_fn: Callable,
    llm=None,
    expanded: bool = False,
    hyde_cache: Optional[Dict[str, str]] = None,
) -> List[Chunk]:
    """Retrieve and deduplicate passages across the principal and variant queries, with optional HyDE and expanded-threshold support."""
    hyde_query: Optional[str] = None
    if llm is not None and USE_HYDE:
        cid = criterion["id"]
        if hyde_cache is not None and cid in hyde_cache:
            hyde_query = hyde_cache[cid]
        else:
            hyde_query = generate_hyde_query(criterion, llm)
            if hyde_cache is not None:
                hyde_cache[cid] = hyde_query

    queries = [criterion["audit_question"]] + criterion.get("query_variants", [])
    # Set k per query so that each variant contributes proportionately to the final pool.
    # With one query, k=10; with five queries, k=2 each, yielding up to ten unique chunks.
    k_per_query = max(2, math.ceil(RETRIEVAL_MAX / len(queries)))

    seen: set = set()
    combined: List[Chunk] = []

    for i, q in enumerate(queries):
        sem_q = hyde_query if i == 0 else None  # Use HyDE only for the principal query.
        for chunk in retrieve_fn(q, semantic_query=sem_q, expanded=expanded, k=k_per_query):
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                combined.append(chunk)

    return combined[:RETRIEVAL_MAX]


def evaluate_criterion(
    criterion: Dict,
    retrieve_fn: Callable,
    llm,
    verdict_scores: Dict[str, Optional[float]],
    hyde_cache: Optional[Dict[str, str]] = None,
) -> Dict:
    retrieved = _u_shape_reorder(_multi_query_retrieve(criterion, retrieve_fn, llm=llm, hyde_cache=hyde_cache))
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", build_user_prompt(criterion, retrieved)),
    ]
    output: CriterionLLMOutput = _structured_invoke(llm, CriterionLLMOutput, messages, retrieved_chunks=retrieved, criterion=criterion)
    score = verdict_scores.get(output.verdict)
    return {
        "criterion_id": criterion["id"],
        "verdict": output.verdict,
        "score": score,
        "reasoning": output.reasoning,
        "evidence": _parse_evidence_json(output.evidence_json),
        "retrieval_chunk_ids": [c.chunk_id for c in retrieved],
        "retrieved_chunks": [
            {"chunk_id": c.chunk_id, "section_title": c.section_title or "", "text": c.text}
            for c in retrieved
        ],
        "retrieval_confidence": {"n_chunks_retrieved": len(retrieved)},
        "verified": False,
        "original_verdict": None,
    }


def query_structured(
    question: str,
    retrieve_fn: Callable,
    llm,
    n_chunks: int = 8,
    use_hyde: bool = True,
) -> Dict:
    """Answer an expert query using structured verdicts, rationales and verbatim evidence for automated evaluation."""
    hyde_query = None
    if use_hyde and USE_HYDE and llm is not None:
        hyde_query = generate_hyde_query_qa(question, llm)

    retrieved = _u_shape_reorder(retrieve_fn(question, semantic_query=hyde_query, k=n_chunks))

    chunks_text = "\n\n".join(
        f"[PASSAGE {i + 1} | chunk_id={c.chunk_id}]\n{c.text}"
        for i, c in enumerate(retrieved)
    )

    messages = [
        ("system", QA_STRUCTURED_SYSTEM_PROMPT),
        ("human", f"Document passages:\n\n{chunks_text}\n\n---\n\nQuestion: {question}"),
    ]

    output: ConsultaLLMOutput = _structured_invoke(
        llm, ConsultaLLMOutput, messages, retrieved_chunks=retrieved
    )

    return {
        "question": question,
        "verdict": output.verdict,
        "reasoning": output.reasoning,
        "evidence": _parse_evidence_json(output.evidence_json),
        "retrieval_chunk_ids": [c.chunk_id for c in retrieved],
        "n_chunks_retrieved": len(retrieved),
    }


def compute_scores(
    results: List[Dict],
    checklist: Dict,
) -> Tuple[List[Dict], Optional[float]]:
    result_map = {r["criterion_id"]: r for r in results}
    category_scores = []
    cat_score_values = []

    for cat in checklist["categories"]:
        cat_id = cat["id"]
        cat_criteria = [c for c in checklist["criteria"] if c["category"] == cat_id]
        evaluated, not_applicable = 0, 0
        weighted_sum = 0.0
        weight_sum = 0.0

        for crit in cat_criteria:
            r = result_map.get(crit["id"])
            if r is None:
                continue
            if r["verdict"] == "NOT_APPLICABLE":
                not_applicable += 1
            else:
                evaluated += 1
                weighted_sum += (r["score"] or 0.0) * crit["weight"]
                weight_sum += crit["weight"]

        cat_score = (weighted_sum / weight_sum) if weight_sum > 0 else None
        category_scores.append(
            {
                "category_id": cat_id,
                "score": round(cat_score, 4) if cat_score is not None else None,
                "criteria_evaluated": evaluated,
                "criteria_not_applicable": not_applicable,
            }
        )
        if cat_score is not None:
            cat_score_values.append(cat_score)

    total_score = (
        round(sum(cat_score_values) / len(cat_score_values), 4)
        if cat_score_values
        else None
    )
    return category_scores, total_score


def run_verification_pass(
    results: List[Dict],
    checklist: Dict,
    retrieve_fn: Callable,
    llm,
    verdict_scores: Dict[str, Optional[float]],
    verbose: bool = True,
    hyde_cache: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Reassess uncertain verdicts with expanded retrieval and retain only changes that indicate stronger documentary coverage."""
    criteria_map = {c["id"]: c for c in checklist["criteria"]}
    verdict_order = {"YES": 3, "PARTIAL": 2, "NOT_STATED": 1, "NOT_APPLICABLE": 0}
    updated: List[Dict] = []

    for result in results:
        if result["verdict"] not in ("NOT_STATED", "PARTIAL"):
            updated.append(result)
            continue

        criterion = criteria_map.get(result["criterion_id"])
        if criterion is None:
            updated.append(result)
            continue

        retrieved = _u_shape_reorder(_multi_query_retrieve(criterion, retrieve_fn, llm=llm, expanded=True, hyde_cache=hyde_cache))
        if not retrieved:
            updated.append(result)
            continue

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", build_user_prompt(criterion, retrieved)),
        ]
        try:
            output: CriterionLLMOutput = _structured_invoke(llm, CriterionLLMOutput, messages, retrieved_chunks=retrieved, criterion=criterion)
        except Exception:
            updated.append(result)
            continue

        new_verdict = output.verdict
        old_verdict = result["verdict"]

        if verdict_order.get(new_verdict, 0) > verdict_order.get(old_verdict, 0):
            if verbose:
                print(f"    ^ {criterion['id']} verified: {old_verdict} -> {new_verdict}")
            updated.append({
                **result,
                "verdict": new_verdict,
                "score": verdict_scores.get(new_verdict),
                "reasoning": output.reasoning,
                "evidence": _parse_evidence_json(output.evidence_json),
                "retrieval_chunk_ids": [c.chunk_id for c in retrieved],
                "retrieved_chunks": [
                    {"chunk_id": c.chunk_id, "section_title": c.section_title or "", "text": c.text}
                    for c in retrieved
                ],
                "retrieval_confidence": {"n_chunks_retrieved": len(retrieved)},
                "verified": True,
                "original_verdict": old_verdict,
            })
        else:
            updated.append(result)

    return updated


def run_audit(
    chunks: List[Chunk],
    retrieve_fn: Callable,
    llm,
    checklist: Dict,
    sector: Optional[str] = None,
    language: Optional[str] = "es",
    captured_at: Optional[str] = None,
    verbose: bool = True,
    run_verification: bool = True,
) -> Dict:
    usage_start = len(llm.model_usage_log) if isinstance(llm, LLMPool) else 0
    verdict_scores: Dict[str, Optional[float]] = checklist["scoring"]["verdicts"]
    results = []
    criteria = checklist["criteria"]
    total = len(criteria)
    hyde_cache: Dict[str, str] = {}  # reuse HyDE output across the initial and verification passes

    for i, criterion in enumerate(criteria, start=1):
        criterion = {**criterion, "_language": language or "es"}
        if verbose:
            print(
                f"  [{i:2d}/{total}] {criterion['id']} · assessing documentary coverage…",
                end=" ",
                flush=True,
            )
        result = evaluate_criterion(criterion, retrieve_fn, llm, verdict_scores, hyde_cache=hyde_cache)
        results.append(result)
        if verbose:
            print(f"{result['verdict']}  (score={result['score']})")

    # Second pass: reassess NOT_STATED and PARTIAL verdicts using expanded retrieval.
    if run_verification:
        uncertain = [r for r in results if r["verdict"] in ("NOT_STATED", "PARTIAL")]
        if uncertain:
            if verbose:
                print(f"\n  [Verification] {len(uncertain)} uncertain criteria -> second pass...")
            results = run_verification_pass(
                results, checklist, retrieve_fn, llm, verdict_scores, verbose, hyde_cache=hyde_cache
            )
            if verbose:
                verified_count = sum(1 for r in results if r.get("verified"))
                print(f"  [Verification] {verified_count} verdicts updated.\n")

    category_scores, total_score = compute_scores(results, checklist)

    return {
        "audit_id": str(uuid.uuid4()),
        "document": {
            "doc_hash": chunks[0].doc_hash,
            "doc_source": chunks[0].doc_source,
            "captured_at": captured_at,
            "language": language,
            "sector": sector,
        },
        "checklist_version": checklist["version"],
        "model_used": _models_used_for_audit(llm, usage_start),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "category_scores": category_scores,
        "total_score": total_score,
        "total_score_pct": round(total_score * 100, 2) if total_score is not None else None,
    }


def validate_audit(audit: Dict, schema: Dict) -> None:
    jsonschema.validate(instance=audit, schema=schema)


def save_audit(audit: Dict, out_dir: Path = AUDITS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_hash_short = audit["document"]["doc_hash"][:16]
    fname = f"{doc_hash_short}_audit.json"
    path = out_dir / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    return path
