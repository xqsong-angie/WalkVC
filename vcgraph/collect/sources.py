"""Source connectors: Y Combinator directory (Crawl4AI), SEC EDGAR Form D, OpenCorporates.

All raw payloads are stored in MongoDB `raw_pages` before any parsing so that the
pipeline is re-playable without re-crawling.
"""
from __future__ import annotations

import asyncio
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ..config import settings
from ..storage import mongo

YC_BASE = "https://www.ycombinator.com"
YC_OSS = "https://yc-oss.github.io/api"  # community mirror of the YC directory (JSON)


# =================================================================== YC (Crawl4AI)
async def crawl_yc_directory(batch: str, max_companies: int = 50, scrolls: int = 6) -> list[str]:
    """Render the (JS-heavy) YC directory with a headless browser and collect company URLs.

    No CSS selectors / site-specific scraping logic: we just let Crawl4AI render the page,
    scroll to trigger lazy loading, and keep internal links that look like /companies/<slug>.
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    url = f"{YC_BASE}/companies?batch={urllib.parse.quote(batch)}"
    scroll_js = ["(async () => { for (let i = 0; i < %d; i++) { window.scrollTo(0, document.body.scrollHeight);"
                 " await new Promise(r => setTimeout(r, 1200)); } })();" % scrolls]
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, js_code=scroll_js,
                               wait_for="css:a[href*='/companies/']", page_timeout=60000,
                               delay_before_return_html=scrolls * 1.3)
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        res = await crawler.arun(url=url, config=run_cfg)
    if not res.success:
        raise RuntimeError(f"YC directory crawl failed: {res.error_message}")
    mongo.save_raw("yc", url, res.html, meta={"batch": batch})

    slug_re = re.compile(r"/companies/([a-z0-9][a-z0-9\-]+)$")
    skip = {"founders", "industry", "location", "batch", "black-founders", "women-founders"}
    urls: list[str] = []
    for link in (res.links or {}).get("internal", []):
        href = link.get("href", "").split("?")[0].rstrip("/")
        m = slug_re.search(href)
        if m and m.group(1) not in skip:
            full = href if href.startswith("http") else YC_BASE + href
            if full not in urls:
                urls.append(full)
    return urls[:max_companies]


async def crawl_pages(urls: list[str], source: str = "yc", concurrency: int = 3) -> list[dict]:
    """Crawl a list of pages; return [{url, markdown}] and persist raw HTML."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=45000)
    out: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        async def one(u: str):
            async with sem:
                r = await crawler.arun(url=u, config=cfg)
                if r.success:
                    md = str(r.markdown or "")
                    mongo.save_raw(source, u, r.html, meta={"markdown": md[:200_000]})
                    out.append({"url": u, "markdown": md})
        await asyncio.gather(*(one(u) for u in urls), return_exceptions=True)
    return out


_SEASONS = {"w": "winter", "s": "summer", "f": "fall", "x": "spring"}


def normalize_batch(batch: str) -> str:
    """'W25' / 'w25' / 'Winter 2025' / 'winter-2025' -> 'Winter 2025'."""
    b = batch.strip().replace("-", " ")
    m = re.fullmatch(r"([WSFXwsfx])\s?(\d{2})", b)
    if m:
        return f"{_SEASONS[m.group(1).lower()].title()} 20{m.group(2)}"
    m = re.fullmatch(r"(winter|summer|fall|spring)\s+(\d{4})", b, re.I)
    return f"{m.group(1).title()} {m.group(2)}" if m else batch


async def fetch_yc_oss(batch: str, limit: int | None = 50) -> list[dict]:
    """Fast path: structured JSON mirror of the YC directory (no browser needed).

    Endpoint: https://yc-oss.github.io/api/batches/<season>-<year>.json
    Note: the mirror has no founder names -> use yc_mode="hybrid" to crawl them.
    """
    slug = normalize_batch(batch).lower().replace(" ", "-")
    url = f"{YC_OSS}/batches/{slug}.json"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(url)
        r.raise_for_status()
    mongo.save_raw("yc", url, r.text, content_type="application/json")
    rows = r.json()
    return rows[:limit] if limit else rows


def yc_oss_to_record(c: dict, batch_hint: str | None = None) -> dict:
    industries = [x for x in (c.get("industries") or []) if x]
    return {
        "source": "yc", "source_id": str(c.get("slug") or c.get("id")), "kind": "company",
        "name": c.get("name", ""),
        "data": {
            "website": c.get("website"), "one_liner": c.get("one_liner"),
            "description": c.get("long_description"),
            "batch": normalize_batch(c.get("batch") or batch_hint or ""),
            "industry": c.get("industry"), "subindustry": c.get("subindustry"),
            "tags": c.get("tags") or [],
            "sectors": sorted({*industries, *([c["subindustry"].split("->")[-1].strip()] if c.get("subindustry") else [])}),
            "location": c.get("all_locations"), "team_size": c.get("team_size"),
            "status": c.get("status"), "stage": c.get("stage"), "launched_at": c.get("launched_at"),
            "yc_url": c.get("url") or f"{YC_BASE}/companies/{c.get('slug')}",
            "founders": [], "investors": [],
        },
    }


# =================================================================== SEC EDGAR
SEC_EFTS = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"


def _sec_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30, headers={"User-Agent": settings.sec_user_agent,
                                                  "Accept-Encoding": "gzip, deflate"})


async def sec_search_form_d(client: httpx.AsyncClient, company: str, max_hits: int = 3) -> list[dict]:
    """Full-text search EDGAR for Form D filings mentioning the company name."""
    params = {"q": f'"{company}"', "forms": "D", "dateRange": "custom", "startdt": "2021-01-01"}
    r = await client.get(SEC_EFTS, params=params)
    await asyncio.sleep(0.15)  # SEC fair-access: <= 10 req/s
    if r.status_code != 200:
        return []
    mongo.save_raw("sec", str(r.url), r.text, content_type="application/json")
    hits = []
    for h in r.json().get("hits", {}).get("hits", [])[:max_hits]:
        src = h.get("_source", {})
        adsh, _, fname = h.get("_id", "").partition(":")
        ciks = src.get("ciks") or []
        if not (adsh and ciks):
            continue
        hits.append({"adsh": adsh, "file": fname or "primary_doc.xml", "cik": ciks[0],
                     "display_name": (src.get("display_names") or [""])[0],
                     "file_date": src.get("file_date")})
    return hits


async def sec_fetch_form_d(client: httpx.AsyncClient, hit: dict) -> dict | None:
    url = f"{SEC_ARCHIVE}/{int(hit['cik'])}/{hit['adsh'].replace('-', '')}/{hit['file']}"
    r = await client.get(url)
    await asyncio.sleep(0.15)
    if r.status_code != 200 or not r.text.lstrip().startswith("<"):
        return None
    mongo.save_raw("sec", url, r.text, content_type="application/xml")
    parsed = parse_form_d(r.text)
    if parsed:
        parsed.update({"accession": hit["adsh"], "cik": hit["cik"], "file_date": hit["file_date"], "url": url})
    return parsed


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _txt(el: ET.Element | None, path: str) -> str | None:
    if el is None:
        return None
    x = el.find(path)
    return x.text.strip() if x is not None and x.text else None


def _securities(od: ET.Element | None) -> list[str]:
    if od is None:
        return []
    t = od.find("typesOfSecuritiesOffered")
    return [] if t is None else [c.tag for c in list(t)]


def parse_form_d(xml_text: str) -> dict | None:
    """Parse Form D primary_doc.xml (issuer, offering amounts, related persons)."""
    try:
        root = _strip_ns(ET.fromstring(xml_text))
    except ET.ParseError:
        return None
    issuer = root.find(".//primaryIssuer")
    if issuer is None:
        return None
    persons = []
    for p in root.findall(".//relatedPersonInfo"):
        first = _txt(p, "relatedPersonName/firstName") or ""
        last = _txt(p, "relatedPersonName/lastName") or ""
        roles = [r.text for r in p.findall(".//relatedPersonRelationshipList/relationship") if r.text]
        if first or last:
            persons.append({"name": f"{first} {last}".strip(), "roles": roles})
    od = root.find(".//offeringData")

    def _num(path):
        v = _txt(od, path)
        try:
            return float(v) if v not in (None, "Indefinite") else None
        except ValueError:
            return None

    return {
        "issuer_name": _txt(issuer, "entityName"),
        "issuer_cik": _txt(issuer, "cik"),
        "state": _txt(issuer, "issuerAddress/stateOrCountry"),
        "city": _txt(issuer, "issuerAddress/city"),
        "jurisdiction": _txt(issuer, "jurisdictionOfInc"),
        "entity_type": _txt(issuer, "entityType"),
        "year_of_inc": _txt(issuer, "yearOfInc/value"),
        "industry_group": _txt(od, "industryGroup/industryGroupType"),
        "date_first_sale": _txt(od, "typeOfFiling/dateOfFirstSale/value"),
        "total_offering_amount": _num("offeringSalesAmounts/totalOfferingAmount"),
        "total_amount_sold": _num("offeringSalesAmounts/totalAmountSold"),
        "num_investors": _num("investors/totalNumberAlreadyInvested"),
        "securities": _securities(od),
        "related_persons": persons,
    }


_SPV = re.compile(r"\ba series of\b|\bspv\b|\bfund\b|\bventures?\b|\bcapital\b|\bpartners\b|,?\s*l\.?p\.?$", re.I)


def issuer_matches(issuer: str | None, company: str) -> bool:
    """EDGAR full-text search is noisy: keep a Form D only if its issuer IS the company
    (same normalized name), and never an SPV / fund that merely invests in it."""
    from .resolve import normalize_name

    if not issuer or _SPV.search(issuer.strip()):
        return False
    a, b = normalize_name(issuer), normalize_name(company)
    return bool(a) and a == b


def filing_plausible(fd: dict, batch: str | None, slack_years: int = 3) -> bool:
    """A same-named issuer whose first sale predates the YC batch by years is another company."""
    by = re.search(r"(\d{4})", batch or "")
    fy = re.match(r"(\d{4})", fd.get("date_first_sale") or fd.get("file_date") or "")
    return not (by and fy) or int(fy.group(1)) >= int(by.group(1)) - slack_years


def form_d_to_record(fd: dict, company_hint: str) -> dict:
    return {"source": "sec", "source_id": fd["accession"], "kind": "filing",
            "name": fd.get("issuer_name") or company_hint, "data": {**fd, "company_hint": company_hint}}


# =================================================================== OpenCorporates
OC_API = "https://api.opencorporates.com/v0.4/companies/search"


async def opencorporates_search(client: httpx.AsyncClient, name: str, max_hits: int = 3) -> list[dict]:
    if not settings.opencorporates_token:
        return []
    r = await client.get(OC_API, params={"q": name, "api_token": settings.opencorporates_token,
                                         "per_page": max_hits, "order": "score"})
    if r.status_code != 200:
        return []
    mongo.save_raw("opencorporates", str(r.url).split("&api_token")[0], r.text, content_type="application/json")
    out = []
    for item in r.json().get("results", {}).get("companies", []):
        c: dict[str, Any] = item.get("company", {})
        out.append({
            "source": "opencorporates",
            "source_id": f"{c.get('jurisdiction_code')}:{c.get('company_number')}",
            "kind": "legal_entity", "name": c.get("name", ""),
            "data": {k: c.get(k) for k in ("jurisdiction_code", "company_number", "incorporation_date",
                                           "current_status", "company_type", "registered_address_in_full",
                                           "opencorporates_url")} | {"company_hint": name},
        })
    return out
