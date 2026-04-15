import os
import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("haqqi.rag")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Haqqi RAG API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# OpenAI async client
# ---------------------------------------------------------------------------
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# BOE constants
# ---------------------------------------------------------------------------
BOE_SEARCH_URL = "https://www.boe.es/buscar/ayudas/legislacion_xml.php"
BOE_DOC_URL = "https://www.boe.es/diario_boe/xml.php"

# Total upper bound well under Vercel's 10 second limit.
# Stage budget: keywords (1.5) + search (2.0) + docs parallel (2.0) + final LLM (4.5)
KEYWORD_TIMEOUT = 1.5          # seconds, very tight for the extractor
BOE_SEARCH_TIMEOUT = 2.0       # seconds
BOE_DOC_TIMEOUT = 2.5          # seconds
OPENAI_TIMEOUT = 5.0           # seconds for the final Darija generation
MAX_DOCS = 2                   # number of BOE documents to retrieve for context
MAX_CONTEXT_CHARS = 6000       # hard cap on context length passed to the LLM

FALLBACK_MESSAGE = (
    "سمح ليا، الخدمة ديال BOE.es كاتعطي مشاكل دابا أو كاتأخر بزاف. "
    "عاود حاول من بعد شوية، و إلا بغيتي جواب سريع حاول تكون بلاصة السؤال ديالك أوضح "
    "(مثلا: حقوق الإقامة، لم الشمل، عقد الكراء...)."
)

KEYWORD_SYSTEM_PROMPT = (
    "You are a Spanish legal search query generator for the BOE (Boletín "
    "Oficial del Estado). The user will ask a question in any language "
    "(often Moroccan Darija in Arabic script). Your only job is to output "
    "2 or 3 PRECISE Spanish legal keywords or short noun phrases that will "
    "maximize recall on https://www.boe.es search. "
    "Rules:\n"
    "- Output Spanish only.\n"
    "- Prefer formal legal terminology (e.g. 'extranjería', 'reagrupación "
    "familiar', 'permiso de residencia', 'arrendamiento urbano', 'contrato "
    "de trabajo', 'Seguridad Social').\n"
    "- 2 or 3 terms maximum, separated by a single space.\n"
    "- No punctuation, no quotes, no explanations, no line breaks.\n"
    "- If the question is generic, pick the most relevant umbrella term."
)

SYSTEM_PROMPT = (
    "انت مساعد قانوني متخصص ف القانون الإسباني، كتخدم مع جالية المغاربة ف إسبانيا. "
    "كتستعمل نظام RAG: غادي نعطيك مقاطع رسمية من الجريدة الرسمية الإسبانية (BOE). "
    "المهمة ديالك:\n"
    "1) قرا المرجع القانوني بالإسبانية مزيان.\n"
    "2) جاوب على السؤال ديال المستخدم **بالدارجة المغربية فقط وبالحروف العربية** "
    "(ماشي فصحى وماشي حروف لاتينية).\n"
    "3) كون دقيق قانونيا، وإيلا المرجع ما كافي قول بصراحة أنه خاص الإنسان يستشير محامي أو "
    "الإدارة المختصة.\n"
    "4) إيلا المرجع ما عندوش علاقة مباشرة بالسؤال، وضّح ذلك وعطي جواب عام مفيد بالدارجة.\n"
    "5) ما تخترعش قوانين أو أرقام مواد إيلا ماكانتش ف المرجع.\n"
    "6) نظم الجواب: شرح قصير، ثم النقط المهمة، ثم نصيحة عملية.\n"
    "ممنوع تجاوب بأي لغة أخرى غير الدارجة المغربية بالحروف العربية."
)


# ---------------------------------------------------------------------------
# Spanish keyword extraction (Step 1 of the RAG pipeline)
# ---------------------------------------------------------------------------
_ASCII_LETTER_RE = None  # set lazily to avoid importing re at module import


def _sanitize_keywords(raw: str) -> str:
    """Clean the LLM output into a BOE-friendly query string."""
    if not raw:
        return ""
    # Strip newlines, quotes, and collapse whitespace.
    cleaned = raw.replace("\n", " ").replace("\r", " ").strip().strip('"').strip("'")
    # Remove trailing punctuation that BOE search doesn't like.
    for ch in ".,;:!?":
        cleaned = cleaned.replace(ch, " ")
    tokens = [t for t in cleaned.split() if t]
    # Keep at most 3 tokens (the prompt asked for 2-3 phrases, but we allow
    # a small safety margin of 5 words in case it returns short phrases).
    tokens = tokens[:5]
    return " ".join(tokens)


def _looks_like_spanish(text: str) -> bool:
    """Cheap heuristic: at least one ASCII letter and no Arabic block chars."""
    has_latin = any("a" <= c.lower() <= "z" for c in text)
    has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)
    return has_latin and not has_arabic


async def extract_spanish_keywords(question: str) -> str:
    """
    Use gpt-4o-mini to turn any-language input into 2-3 Spanish legal keywords.

    Returns the fallback (the raw question) only if extraction fails; the
    caller can still pass that to BOE, though results will be weaker.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=24,
            messages=[
                {"role": "system", "content": KEYWORD_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            timeout=KEYWORD_TIMEOUT,
        )
    except Exception as exc:
        logger.warning("Keyword extraction failed: %s", exc)
        return question

    keywords = _sanitize_keywords(response.choices[0].message.content or "")
    if not keywords or not _looks_like_spanish(keywords):
        logger.info("Keyword extraction produced unusable output: %r", keywords)
        return question
    logger.info("Extracted Spanish keywords: %s", keywords)
    return keywords


# ---------------------------------------------------------------------------
# BOE retrieval helpers
# ---------------------------------------------------------------------------
def _parse_search_ids(xml_bytes: bytes, limit: int = MAX_DOCS) -> List[str]:
    """Extract up to `limit` BOE identifiers from a legislation search XML."""
    ids: List[str] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("BOE search XML parse error: %s", exc)
        return ids

    # The search endpoint returns <id> elements scattered across result nodes.
    for node in root.iter("id"):
        if node.text and node.text.strip():
            ids.append(node.text.strip())
            if len(ids) >= limit:
                break
    return ids


def _parse_document_text(xml_bytes: bytes) -> str:
    """Extract the plain legal text from a BOE document XML payload."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("BOE document XML parse error: %s", exc)
        return ""

    parts: List[str] = []
    # BOE documents expose the full text in <texto> blocks with <p> paragraphs.
    for texto in root.iter("texto"):
        # Gather all text children (p, h3, etc.) flattened.
        for child in texto.iter():
            if child.text and child.text.strip():
                parts.append(child.text.strip())
    # Fallback: if no <texto> blocks, try the document title/summary.
    if not parts:
        for tag in ("titulo", "materia", "sumario"):
            for node in root.iter(tag):
                if node.text and node.text.strip():
                    parts.append(node.text.strip())
    return "\n".join(parts)


async def _fetch_boe_document(
    http_client: httpx.AsyncClient, boe_id: str
) -> Tuple[str, str]:
    """Return (boe_id, text) for a single document. Empty text on failure."""
    try:
        res = await http_client.get(
            BOE_DOC_URL, params={"id": boe_id}, timeout=BOE_DOC_TIMEOUT
        )
        if res.status_code != 200:
            logger.info("BOE doc %s returned HTTP %s", boe_id, res.status_code)
            return boe_id, ""
        return boe_id, _parse_document_text(res.content)
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.warning("BOE document fetch failed for %s: %s", boe_id, exc)
        return boe_id, ""


async def retrieve_boe_context(query: str) -> Tuple[str, List[str]]:
    """
    Retrieve relevant legal context from the BOE.

    Returns (context_text, [boe_ids]). If BOE is unreachable or returns no
    useful data, context_text will be an empty string.
    """
    sources: List[str] = []
    async with httpx.AsyncClient(
        headers={
            "User-Agent": "HaqqiApp/2.0 (+https://boe.es legal assistant)",
            "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
        },
        follow_redirects=True,
    ) as http_client:
        # 1. Search for relevant legislation IDs.
        try:
            search_res = await http_client.get(
                BOE_SEARCH_URL,
                params={"txt": query},
                timeout=BOE_SEARCH_TIMEOUT,
            )
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.warning("BOE search request failed: %s", exc)
            return "", sources

        if search_res.status_code != 200:
            logger.info("BOE search returned HTTP %s", search_res.status_code)
            return "", sources

        ids = _parse_search_ids(search_res.content, limit=MAX_DOCS)
        if not ids:
            return "", sources

        # 2. Fetch the top documents in parallel to save wall-clock time.
        results = await asyncio.gather(
            *[_fetch_boe_document(http_client, boe_id) for boe_id in ids],
            return_exceptions=False,
        )

    # 3. Stitch a compact context, keeping track of IDs that actually produced text.
    chunks: List[str] = []
    total = 0
    for boe_id, text in results:
        if not text:
            continue
        sources.append(boe_id)
        # Trim each document so one huge file doesn't eat the whole budget.
        per_doc_budget = max(1000, MAX_CONTEXT_CHARS // max(1, len(results)))
        snippet = text[:per_doc_budget]
        chunk = f"[Fuente BOE: {boe_id}]\n{snippet}"
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_CONTEXT_CHARS:
            break

    return "\n\n---\n\n".join(chunks)[:MAX_CONTEXT_CHARS], sources


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
async def generate_darija_answer(question: str, context: str) -> str:
    """Call OpenAI to produce a Moroccan Darija answer grounded on BOE context."""
    if context:
        user_content = (
            f"المرجع القانوني من BOE.es (بالإسبانية):\n{context}\n\n"
            f"السؤال ديال المستخدم:\n{question}\n\n"
            "جاوب بالدارجة المغربية بالحروف العربية فقط، "
            "و اعتمد على المرجع اللي فوق."
        )
    else:
        user_content = (
            "ما لقيتش مرجع مباشر ف BOE.es على هاد السؤال.\n"
            f"السؤال: {question}\n\n"
            "عطي جواب عام مفيد بالدارجة المغربية بالحروف العربية، "
            "ونصح المستخدم يتأكد من محامي أو الإدارة المختصة."
        )

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        timeout=OPENAI_TIMEOUT,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
@app.get("/api/index/ask")
async def ask(question: str):
    question = (question or "").strip()
    if not question:
        return {
            "answer": "كتب سؤالك عافاك باش نقدر نعاونك.",
            "sources": [],
            "search_query": "",
        }

    # 1. Extract Spanish legal keywords (fast, ~1.5s budget).
    try:
        search_query = await asyncio.wait_for(
            extract_spanish_keywords(question), timeout=KEYWORD_TIMEOUT + 0.2
        )
    except asyncio.TimeoutError:
        logger.warning("Keyword extraction timed out for query: %s", question)
        search_query = question  # graceful degradation

    # 2. Retrieve BOE context using the Spanish keywords.
    try:
        context, sources = await asyncio.wait_for(
            retrieve_boe_context(search_query),
            timeout=BOE_SEARCH_TIMEOUT + BOE_DOC_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("BOE retrieval exceeded global budget for query: %s", search_query)
        context, sources = "", []

    # 3. Generate the Darija answer, with a fallback if the LLM call fails.
    try:
        answer = await asyncio.wait_for(
            generate_darija_answer(question, context), timeout=OPENAI_TIMEOUT + 1.0
        )
    except asyncio.TimeoutError:
        logger.warning("OpenAI generation timed out for query: %s", question)
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": sources,
            "search_query": search_query,
        }
    except Exception as exc:  # openai errors, network, etc.
        logger.exception("OpenAI generation failed: %s", exc)
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": sources,
            "search_query": search_query,
        }

    return {"answer": answer, "sources": sources, "search_query": search_query}


@app.get("/api/index/health")
async def health():
    return {"status": "ok"}
