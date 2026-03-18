"""Crawler Engine — Phase 1.

- async httpx fetcher with subprocess curl fallback
- trafilatura for clean text extraction
- BeautifulSoup for raw link extraction (not scored yet)
- robots.txt cache + per-domain 1 s delay
"""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import warnings

import trafilatura
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


_HEADERS = {
    "User-Agent": "Tipster/0.1 (+https://github.com/tipster; research crawler)",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

_ROBOTS_CACHE: dict[str, tuple[RobotFileParser, float]] = {}
_DOMAIN_LAST_FETCH: dict[str, float] = {}
_ROBOTS_TTL = 3600  # re-fetch robots.txt after 1 hour


@dataclass
class CrawlResult:
    url: str
    status_code: int
    raw_html: str = ""
    text: str = ""             # trafilatura-extracted plain text
    links: list[str] = field(default_factory=list)           # outbound URLs (deduped)
    link_data: list[tuple[str, str]] = field(default_factory=list)  # (url, anchor_text) pairs
    content_hash: str = ""     # SHA-256 of normalised text
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status_code < 400 and self.error is None

    @property
    def inaccessible(self) -> bool:
        return self.status_code in (401, 403, 404, 410, 451) or self.status_code >= 500


def _sha256(text: str) -> str:
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _extract_links(html: str, base_url: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (urls, link_data) where link_data is (url, anchor_text) pairs."""
    soup = BeautifulSoup(html, "html.parser")
    seen: dict[str, str] = {}  # url → anchor_text (first occurrence wins)
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme in ("http", "https"):
            normalised = absolute.split("#")[0]
            if normalised not in seen:
                anchor = tag.get_text(strip=True)[:200]
                seen[normalised] = anchor
    urls = list(seen.keys())
    link_data = list(seen.items())
    return urls, link_data


async def _check_robots(url: str, delay: float) -> bool:
    """Return True if the URL is allowed by robots.txt."""
    parsed = urlparse(url)
    domain_key = f"{parsed.scheme}://{parsed.netloc}"
    now = time.time()

    cached = _ROBOTS_CACHE.get(domain_key)
    if cached is None or (now - cached[1]) > _ROBOTS_TTL:
        robots_url = f"{domain_key}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, rp.read)
        except Exception:
            pass  # treat as allowed on error
        _ROBOTS_CACHE[domain_key] = (rp, now)
    else:
        rp = cached[0]

    return rp.can_fetch(_HEADERS["User-Agent"], url)


async def _domain_delay(domain: str, delay: float) -> None:
    """Enforce minimum inter-request delay per domain."""
    last = _DOMAIN_LAST_FETCH.get(domain, 0.0)
    elapsed = time.time() - last
    if elapsed < delay:
        await asyncio.sleep(delay - elapsed)
    _DOMAIN_LAST_FETCH[domain] = time.time()


async def fetch(url: str, default_delay: float = 1.0, timeout: float = 30.0) -> CrawlResult:
    """Fetch a URL and return a CrawlResult.

    Uses httpx async client; falls back to subprocess curl on connection error.
    """
    parsed = urlparse(url)
    domain = parsed.netloc

    # robots.txt check
    allowed = await _check_robots(url, default_delay)
    if not allowed:
        return CrawlResult(url=url, status_code=403, error="blocked by robots.txt")

    # Rate limit
    await _domain_delay(domain, default_delay)

    # --- httpx attempt ---
    html = ""
    status = 0
    error: Optional[str] = None

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            resp = await client.get(url)
            status = resp.status_code
            if resp.status_code < 400:
                html = resp.text
    except (httpx.ConnectError, httpx.TimeoutException, httpx.TooManyRedirects) as exc:
        # curl fallback
        html, status, error = await _curl_fallback(url, timeout)
    except Exception as exc:
        error = str(exc)
        return CrawlResult(url=url, status_code=0, error=error)

    if status >= 400:
        return CrawlResult(url=url, status_code=status)

    # Extract page content as Markdown.
    # trafilatura is preferred for article pages (it strips boilerplate cleanly);
    # markdownify is used as a fallback for structured/list pages that trafilatura
    # tends to strip too aggressively (e.g. GitHub trending, search result pages).
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        output_format="markdown",
    ) or ""

    if len(text) < 200 and html:
        # trafilatura found little content — likely a structured/list page.
        # Fall back to markdownify on the cleaned body for better structure
        # preservation (headers, lists, tables all become readable Markdown).
        try:
            from markdownify import markdownify as _md
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "iframe",
                              "nav", "footer", "header", "aside"]):
                tag.decompose()
            body = soup.find("body") or soup
            text = _md(str(body), heading_style="ATX", newline_style="backslash") or ""
            # Collapse excessive blank lines
            import re as _re
            text = _re.sub(r"\n{3,}", "\n\n", text).strip()
        except Exception:
            pass  # keep whatever trafilatura gave us

    # Extract outbound links
    links, link_data = _extract_links(html, url)

    content_hash = _sha256(text) if text else _sha256(html[:2000])

    return CrawlResult(
        url=url,
        status_code=status,
        raw_html=html,
        text=text,
        links=links,
        link_data=link_data,
        content_hash=content_hash,
    )


async def _curl_fallback(url: str, timeout: float) -> tuple[str, int, Optional[str]]:
    """Run curl in a subprocess as fallback fetcher."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [
                    "curl", "-sL",
                    "--max-time", str(int(timeout)),
                    "--user-agent", _HEADERS["User-Agent"],
                    "-w", "\n__STATUS_CODE__%{http_code}",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            ),
        )
        output = result.stdout
        if "__STATUS_CODE__" in output:
            parts = output.rsplit("__STATUS_CODE__", 1)
            html = parts[0]
            try:
                status = int(parts[1].strip())
            except ValueError:
                status = 200
        else:
            html = output
            status = 200 if output else 0
        return html, status, None
    except Exception as exc:
        return "", 0, str(exc)
