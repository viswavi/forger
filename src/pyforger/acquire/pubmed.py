"""PubMed retrieval through NCBI's E-utilities (esearch, then efetch).

PubMed holds citations and abstracts: journal articles, case reports, reviews,
trials, letters. It has no full text; ``pmcid`` names the PubMed Central copy
when there is one.

NCBI allows 3 requests per second without an API key and 10 with one. Every
request in the process goes through one shared limiter (2.5 or 8 per second),
and a rate-limited (429) or failed (5xx, connection) request is retried with
backoff; a 429 also holds back the other threads' requests.
"""
from __future__ import annotations

import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from pyforger.acquire.sources import RetrievalSource, RetrieveMode
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("Retrieval Sources")

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class _RateLimiter:
    """Space requests at least ``interval`` seconds apart, across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self, interval: float) -> None:
        """Block until this request's slot, then reserve the next one."""
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + interval
        if start > now:
            time.sleep(start - now)

    def pause(self, seconds: float) -> None:
        """Hold every thread's next request for ``seconds`` (after a 429)."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


# One limiter for the whole process: NCBI counts requests per IP (or per key),
# not per source object.
_LIMITER = _RateLimiter()


def _text(el: ET.Element | None) -> str:
    """All text inside an element, inline markup (<i>, <sup>) included."""
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _abstract(parent: ET.Element | None) -> str:
    """Abstract sections joined by blank lines, each led by its label if any.

    E.g. "BACKGROUND: ..." and "CASE PRESENTATION: ..." as two paragraphs.
    """
    if parent is None:
        return ""
    parts = []
    for section in parent.findall("Abstract/AbstractText"):
        body = _text(section)
        if not body:
            continue
        label = section.get("Label")
        parts.append(f"{label}: {body}" if label else body)
    return "\n\n".join(parts)


def _authors(parent: ET.Element | None) -> list[str]:
    """Author names as "Last Initials" (e.g. "Smith JA"), or a group's name."""
    if parent is None:
        return []
    names = []
    for author in parent.findall("AuthorList/Author"):
        last = author.findtext("LastName")
        if last:
            initials = author.findtext("Initials")
            names.append(f"{last} {initials}" if initials else last)
        elif author.findtext("CollectiveName"):
            names.append(_text(author.find("CollectiveName")))
    return names


def _year(*dates: ET.Element | None) -> str:
    """The first four-digit year in the given date elements, else ""."""
    for date in dates:
        if date is None:
            continue
        year = date.findtext("Year")
        if year:
            return year
        match = re.search(r"\b(\d{4})\b", date.findtext("MedlineDate") or "")
        if match:
            return match.group(1)
    return ""


class PubMedRetrievalSource(RetrievalSource):
    """Search PubMed and return each hit's citation and abstract.

    QUERY mode only. The query uses PubMed's search syntax, so plain words
    work ("lung cancer case reports" matches the Case Reports publication
    type) and so do field tags (``"lung neoplasms"[MeSH] AND case reports[pt]``).
    Results come in PubMed's relevance order unless ``sort=`` says otherwise
    (``"pub_date"``, ``"first_author"``, ``"journal"``).

    Rows have ``text`` (title, then abstract), ``id`` (the PMID), ``title``,
    ``summary`` (the abstract, "" when PubMed has none), ``authors``
    (e.g. ["Smith JA", "Lee K"]), ``journal``, ``year``,
    ``publication_types`` (e.g. ["Case Reports", "Journal Article"]),
    ``mesh_terms``, ``doi``, ``pmcid`` and ``url``.
    """

    _SEARCH_CAP = 10_000  # esearch returns at most the first 10,000 PMIDs
    _SEARCH_PAGE = 10_000
    _FETCH_BATCH = 200  # NCBI asks for POST and batches of a few hundred
    _DEFAULT_TIMEOUT = 30
    _MAX_RETRIES = 5
    # NCBI allows 3 (10 with a key) requests per second. Requests spaced at
    # exactly 1/3 s still drew 429s in a live test (6 of 30), since network
    # delay bunches them up on arrival; these leave a margin.
    _INTERVAL_NO_KEY = 0.4  # 2.5 requests per second
    _INTERVAL_KEY = 0.125  # 8 requests per second

    def cache_key(self) -> tuple:
        """Identify the configuration; the API key changes speed, not results."""
        return (type(self).__module__, type(self).__qualname__, self._kwargs)

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Store options for PubMed retrieval.

        Args:
            api_key: NCBI API key; defaults to ``NCBI_API_KEY``. Optional: it
                raises the rate limit from 3 to 10 requests per second.
            **kwargs: ``timeout`` (seconds per request, default 30) and
                fetch overrides such as ``sort="pub_date"``.
        """
        self._api_key = api_key or os.environ.get("NCBI_API_KEY") or None
        self._kwargs = kwargs
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Return the shared requests.Session, created once."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": "pyforger/0.1 (https://github.com/viswavi/forger)"}
            )
        return self._session

    def _request(
        self, endpoint: str, params: dict[str, Any], merged: dict[str, Any]
    ) -> requests.Response:
        """Send one rate-limited E-utilities request, retrying 429s and 5xx.

        Waits ``Retry-After`` when NCBI sends it, else 1, 2, 4, 8, 16 seconds.
        efetch goes as a POST, so a long PMID list stays out of the URL.
        """
        params = {**params, "tool": "pyforger"}
        if self._api_key:
            params["api_key"] = self._api_key
        interval = self._INTERVAL_KEY if self._api_key else self._INTERVAL_NO_KEY
        timeout = merged.get("timeout", self._DEFAULT_TIMEOUT)
        session = self._get_session()
        url = _EUTILS + endpoint
        for attempt in range(self._MAX_RETRIES + 1):
            _LIMITER.wait(interval)
            delay = float(2**attempt)
            try:
                if endpoint == "efetch.fcgi":
                    resp = session.post(url, data=params, timeout=timeout)
                else:
                    resp = session.get(url, params=params, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self._MAX_RETRIES:
                    raise
                reason = type(exc).__name__
            else:
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable or attempt == self._MAX_RETRIES:
                    resp.raise_for_status()
                    return resp
                reason = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                if resp.status_code == 429:
                    _LIMITER.pause(delay)  # other threads would hit it too
            logger.warning(
                "PubMed %s: %s; retrying in %.0f s (attempt %d of %d).",
                endpoint,
                reason,
                delay,
                attempt + 1,
                self._MAX_RETRIES,
            )
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _search(self, query: str, n: int, merged: dict[str, Any]) -> list[str]:
        """The first ``n`` PMIDs for ``query``, in the requested sort order."""
        pmids: list[str] = []
        while len(pmids) < n:
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retstart": len(pmids),
                "retmax": min(n - len(pmids), self._SEARCH_PAGE),
                "sort": merged.get("sort", "relevance"),
            }
            result = self._request("esearch.fcgi", params, merged).json()
            if "error" in result:
                raise RuntimeError(f"PubMed search failed: {result['error']}")
            page = result.get("esearchresult", {}).get("idlist", [])
            if not page:
                break
            pmids.extend(page)
        return pmids[:n]

    @staticmethod
    def _article_to_item(article: ET.Element) -> dict[str, Any] | None:
        """Convert one <PubmedArticle> or <PubmedBookArticle> to a row."""
        if article.tag == "PubmedArticle":
            citation = article.find("MedlineCitation")
            if citation is None:
                return None
            pmid = citation.findtext("PMID") or ""
            body = citation.find("Article")
            title = _text(body.find("ArticleTitle")) if body is not None else ""
            summary = _abstract(body)
            authors = _authors(body)
            journal = _text(body.find("Journal/Title")) if body is not None else ""
            year = _year(
                body.find("Journal/JournalIssue/PubDate") if body is not None else None,
                body.find("ArticleDate") if body is not None else None,
            )
            types = (
                [_text(t) for t in body.findall("PublicationTypeList/PublicationType")]
                if body is not None
                else []
            )
            mesh = [
                _text(d)
                for d in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
            ]
        elif article.tag == "PubmedBookArticle":
            book = article.find("BookDocument")
            if book is None:
                return None
            pmid = book.findtext("PMID") or ""
            title = _text(book.find("ArticleTitle")) or _text(
                book.find("Book/BookTitle")
            )
            summary = _abstract(book)
            authors = _authors(book)
            journal = _text(book.find("Book/BookTitle"))
            year = _year(book.find("Book/PubDate"))
            types = [_text(t) for t in book.findall("PublicationType")]
            mesh = []
        else:
            return None
        ids = {
            el.get("IdType"): (el.text or "").strip()
            for el in article.findall("PubmedData/ArticleIdList/ArticleId")
        }
        return {
            "text": f"{title}\n\n{summary}".strip(),
            "id": pmid,
            "title": title,
            "summary": summary,
            "authors": authors,
            "journal": journal,
            "year": year,
            "publication_types": types,
            "mesh_terms": mesh,
            "doi": ids.get("doi", ""),
            "pmcid": ids.get("pmc", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

    def _fetch_records(self, pmids: list[str], merged: dict[str, Any]) -> list[dict]:
        """Rows for ``pmids``, in the same order; PMIDs PubMed omits are dropped."""
        by_pmid: dict[str, dict[str, Any]] = {}
        for i in range(0, len(pmids), self._FETCH_BATCH):
            batch = pmids[i : i + self._FETCH_BATCH]
            params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            root = ET.fromstring(self._request("efetch.fcgi", params, merged).content)
            for article in root:
                item = self._article_to_item(article)
                if item is not None and item["id"]:
                    by_pmid[item["id"]] = item
        return [by_pmid[p] for p in pmids if p in by_pmid]

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the top ``n`` PubMed records for ``query`` (QUERY mode only).

        Args:
            mode: Must be ``RetrieveMode.QUERY``.
            query: PubMed search string; required.
            n: Number of records, at most 10,000 (esearch's limit).
            seed: Unused; PubMed search is deterministic for a fixed index.
            **kwargs: Overrides such as ``sort`` and ``timeout``.

        Returns:
            One row per record, in search order (see the class docstring).
        """
        if mode != RetrieveMode.QUERY:
            raise ValueError("PubMed supports QUERY mode only.")
        if not query:
            raise ValueError("PubMed needs a query, e.g. 'lung cancer case reports'.")
        if n > self._SEARCH_CAP:
            raise ValueError(
                f"n={n} exceeds PubMed search's limit of {self._SEARCH_CAP} results."
            )
        if n <= 0:
            return []
        merged = {**self._kwargs, **kwargs}
        pmids = self._search(query, n, merged)
        return self._fetch_records(pmids, merged) if pmids else []
