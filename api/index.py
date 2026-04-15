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
app = FastAPI(title="Haqqi RAG API", version="2.2.0")
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
# Stage budget: keywords (1.5) + search (5.0) + docs parallel (2.0) + final LLM (4.0)
KEYWORD_TIMEOUT = 1.5          # seconds, very tight for the extractor
BOE_SEARCH_TIMEOUT = 5.0       # seconds — bumped per user request
BOE_DOC_TIMEOUT = 2.5          # seconds (runs in parallel for all docs)
OPENAI_TIMEOUT = 5.0           # seconds for the final Darija generation
MAX_DOCS = 2                   # number of BOE documents to retrieve for context
MAX_CONTEXT_CHARS = 6000       # hard cap on context length passed to the LLM

FALLBACK_MESSAGE = (
    "سمح ليا، الخدمة ديال BOE.es كاتعطي مشاكل دابا أو كاتأخر بزاف. "
    "عاود حاول من بعد شوية، و إلا بغيتي جواب سريع حاول تكون بلاصة السؤال ديالك أوضح "
    "(مثلا: حقوق الإقامة، لم الشمل، عقد الكراء...)."
)

# Sentinel returned to the client whenever BOE produced zero usable docs.
NO_LAW_FOUND_MESSAGE = (
    "I couldn't find a specific law in the BOE database for this query."
)

KEYWORD_SYSTEM_PROMPT = (
    "You are an expert Spanish-law search-query generator for the BOE "
    "(Boletín Oficial del Estado). Users ask in any language — Moroccan "
    "Darija, Arabic, French, English, Spanish — and often use slang, "
    "colloquial words, or vague phrasing. Your ONLY job is to translate the "
    "intent into 2-3 FORMAL Spanish legal search terms that will retrieve "
    "the right law or royal decree from https://www.boe.es.\n\n"
    "MANDATORY RULES:\n"
    "- Output Spanish ONLY. Never Arabic, never English, never French.\n"
    "- Always use the exact technical Spanish legal term, not a literal "
    "  translation. Examples of correct mappings:\n"
    "    • residence / الإقامة / carte de séjour → 'permiso de residencia' "
    "or 'autorización de residencia'\n"
    "    • family reunification / لم الشمل → 'reagrupación familiar'\n"
    "    • rent contract / كراء → 'arrendamiento urbano' or 'Ley de "
    "Arrendamientos Urbanos'\n"
    "    • work contract / خدمة → 'contrato de trabajo' or 'Estatuto de los "
    "Trabajadores'\n"
    "    • dismissal / طرد من الخدمة → 'despido improcedente'\n"
    "    • minimum wage / الأجر الأدنى → 'salario mínimo interprofesional'\n"
    "    • social security / ضمان اجتماعي → 'Seguridad Social'\n"
    "    • nationality / جنسية → 'nacionalidad española'\n"
    "    • asylum / لجوء → 'protección internacional' or 'asilo'\n"
    "    • divorce / طلاق → 'divorcio' or 'Código Civil divorcio'\n"
    "    • driving license / رخصة السياقة → 'permiso de conducción'\n"
    "    • taxes / ضرائب → 'Ley General Tributaria' or 'IRPF'\n"
    "    • deportation / ترحيل → 'expulsión extranjero' or 'Ley de "
    "Extranjería'\n"
    "- If the question is about immigrants, add 'extranjería' when "
    "appropriate.\n"
    "- 2 or 3 terms maximum, separated by a single space.\n"
    "- No punctuation, no quotes, no explanations, no line breaks, no "
    "conjunctions like 'y', 'de', 'la' unless part of a legal name.\n"
    "- If totally unclear, output the single best umbrella term "
    "(e.g. 'extranjería' for immigrant life questions).\n"
    "- Never output English, never refuse, never ask back. Just output the "
    "terms."
)

SYSTEM_PROMPT = (
    "انت مساعد قانوني متخصص ف القانون الإسباني، كتخدم مع جالية المغاربة ف إسبانيا. "
    "كتستعمل نظام RAG: غادي نعطيك مقاطع رسمية من الجريدة الرسمية الإسبانية (BOE). "
    "قواعد إجبارية:\n"
    "1) قرا المرجع القانوني بالإسبانية مزيان.\n"
    "2) جاوب **بالدارجة المغربية فقط وبالحروف العربية** (ماشي فصحى، ماشي حروف لاتينية).\n"
    "3) **واجب تذكر الـ BOE ID ولا رقم القانون/المرسوم** اللي خذيتي منو "
    "المعلومة (مثلا: 'حسب BOE-A-2000-544' ولا 'حسب Ley Orgánica 4/2000').\n"
    "4) اعتمد حصرياً على المرجع اللي عطيتك. ممنوع تخترع قوانين ولا أرقام مواد.\n"
    "5) إيلا المرجع ما فيهش جواب دقيق على السؤال، قول بصراحة بأن المرجع "
    "ما كيعطيش التفصيل الكافي ونصح يستشير محامي.\n"
    "6) نظم الجواب: شرح قصير، النقط المهمة (مع الـ BOE ID)، ثم نصيحة عملية.\n"
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


def _count_search_ids(xml_bytes: bytes) -> int:
    """Count the total number of <id> entries in a BOE search XML response."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return 0
    return sum(1 for n in root.iter("id") if n.text and n.text.strip())


async def retrieve_boe_context(query: str) -> Tuple[str, List[str], int]:
    """
    Retrieve relevant legal context from the BOE.

    Returns (context_text, [boe_ids_with_text], total_ids_found).
    - total_ids_found is the number of <id> entries the BOE search returned,
      regardless of whether we then managed to fetch their document body.
    - context_text is empty when BOE is unreachable or no docs produced text.
    """
    sources: List[str] = []
    total_found = 0

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "HaqqiApp/2.2 (+https://boe.es legal assistant)",
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
            logger.warning("BOE search request failed for %r: %s", query, exc)
            return "", sources, 0

        if search_res.status_code != 200:
            logger.info(
                "BOE search returned HTTP %s for %r",
                search_res.status_code,
                query,
            )
            return "", sources, 0

        total_found = _count_search_ids(search_res.content)
        ids = _parse_search_ids(search_res.content, limit=MAX_DOCS)
        logger.info(
            "BOE search for %r: total_found=%d, fetching=%d",
            query,
            total_found,
            len(ids),
        )
        if not ids:
            return "", sources, total_found

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

    return "\n\n---\n\n".join(chunks)[:MAX_CONTEXT_CHARS], sources, total_found


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
async def generate_darija_answer(question: str, context: str, sources: List[str]) -> str:
    """
    Generate a Moroccan Darija answer grounded on BOE context.

    Precondition: `context` is non-empty. Callers must handle the no-context
    case themselves (typically by returning NO_LAW_FOUND_MESSAGE).
    """
    sources_hint = ", ".join(sources) if sources else "(ما كاين حتى BOE ID)"
    user_content = (
        f"المرجع القانوني من BOE.es (بالإسبانية):\n{context}\n\n"
        f"الـ BOE IDs المتوفرة: {sources_hint}\n\n"
        f"السؤال ديال المستخدم:\n{question}\n\n"
        "جاوب بالدارجة المغربية بالحروف العربية فقط. "
        "**واجب تذكر الـ BOE ID ولا رقم القانون** اللي جا منو الجواب."
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
            "debug": {"docs_found": 0, "docs_used": 0},
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
        context, sources, docs_found = await asyncio.wait_for(
            retrieve_boe_context(search_query),
            timeout=BOE_SEARCH_TIMEOUT + BOE_DOC_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "BOE retrieval exceeded global budget for query: %s", search_query
        )
        context, sources, docs_found = "", [], 0

    debug_payload = {
        "docs_found": docs_found,      # total <id>s returned by BOE search
        "docs_used": len(sources),     # how many actually had usable text
    }

    # 3a. No BOE data → return the exact sentinel. Do NOT let the LLM make
    #     up a general answer, per product requirement.
    if not context:
        logger.info(
            "No BOE context for query=%r search_query=%r (docs_found=%d)",
            question,
            search_query,
            docs_found,
        )
        return {
            "answer": NO_LAW_FOUND_MESSAGE,
            "sources": sources,
            "search_query": search_query,
            "debug": debug_payload,
        }

    # 3b. BOE returned context → force the LLM to ground on it and cite BOE IDs.
    try:
        answer = await asyncio.wait_for(
            generate_darija_answer(question, context, sources),
            timeout=OPENAI_TIMEOUT + 1.0,
        )
    except asyncio.TimeoutError:
        logger.warning("OpenAI generation timed out for query: %s", question)
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": sources,
            "search_query": search_query,
            "debug": debug_payload,
        }
    except Exception as exc:  # openai errors, network, etc.
        logger.exception("OpenAI generation failed: %s", exc)
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": sources,
            "search_query": search_query,
            "debug": debug_payload,
        }

    return {
        "answer": answer,
        "sources": sources,
        "search_query": search_query,
        "debug": debug_payload,
    }


@app.get("/api/index/health")
async def health():
    return {"status": "ok"}
