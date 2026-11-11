"""Tests for PubMedRetrievalSource and retrieve_from_pubmed with mocked E-utilities."""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from pyforger.acquire import PubMedRetrievalSource, RetrieveMode
from pyforger.acquire import pubmed as pubmed_mod
from pyforger.acquire import retrieve_from_pubmed

ARTICLE = """
<PubmedArticle>
  <MedlineCitation>
    <PMID>111</PMID>
    <Article>
      <Journal><Title>BMC pulmonary medicine</Title>
        <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>Solitary <i>multicystic</i> lung cancer: two case reports.
      </ArticleTitle>
      <Abstract>
        <AbstractText Label="BACKGROUND">Rare disease.</AbstractText>
        <AbstractText Label="CASE PRESENTATION">Two  elderly men.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>Tang</LastName><Initials>X</Initials></Author>
        <Author><CollectiveName>Lung Study Group</CollectiveName></Author>
      </AuthorList>
      <PublicationTypeList>
        <PublicationType>Case Reports</PublicationType>
        <PublicationType>Journal Article</PublicationType>
      </PublicationTypeList>
    </Article>
    <MeshHeadingList>
      <MeshHeading><DescriptorName>Lung Neoplasms</DescriptorName></MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
  <PubmedData><ArticleIdList>
    <ArticleId IdType="pubmed">111</ArticleId>
    <ArticleId IdType="doi">10.1186/x</ArticleId>
    <ArticleId IdType="pmc">PMC8591847</ArticleId>
  </ArticleIdList></PubmedData>
</PubmedArticle>
"""

LETTER = """
<PubmedArticle>
  <MedlineCitation>
    <PMID>222</PMID>
    <Article>
      <Journal><Title>Asian journal of surgery</Title>
        <JournalIssue><PubDate><MedlineDate>2024 May-Jun</MedlineDate></PubDate>
        </JournalIssue></Journal>
      <ArticleTitle>A rare case of translobar lung cancer.</ArticleTitle>
      <PublicationTypeList><PublicationType>Letter</PublicationType></PublicationTypeList>
    </Article>
  </MedlineCitation>
</PubmedArticle>
"""

BOOK = """
<PubmedBookArticle>
  <BookDocument>
    <PMID>333</PMID>
    <Book><BookTitle>StatPearls</BookTitle><PubDate><Year>2023</Year></PubDate></Book>
    <ArticleTitle>Lung Cancer</ArticleTitle>
    <Abstract><AbstractText>Overview of lung cancer.</AbstractText></Abstract>
    <PublicationType>Review</PublicationType>
  </BookDocument>
</PubmedBookArticle>
"""


def _xml(*articles: str) -> bytes:
    """Wrap articles in a <PubmedArticleSet>."""
    return f"<PubmedArticleSet>{''.join(articles)}</PubmedArticleSet>".encode()


def _resp(status: int = 200, *, json: Any = None, content: bytes = b"", headers=None):
    """A mocked requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json
    resp.content = content
    resp.headers = headers or {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    return resp


def _search(*pmids: str) -> MagicMock:
    """An esearch JSON response listing ``pmids``."""
    return _resp(json={"esearchresult": {"idlist": list(pmids)}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """No real waiting: the limiter and the backoff both call time.sleep."""
    sleeps: list[float] = []
    monkeypatch.setattr(pubmed_mod.time, "sleep", sleeps.append)
    monkeypatch.setattr(pubmed_mod, "_LIMITER", pubmed_mod._RateLimiter())
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    return sleeps


@pytest.fixture
def session():
    """A mocked requests.Session for every PubMed source."""
    with patch("pyforger.acquire.pubmed.requests.Session") as cls:
        sess = MagicMock()
        cls.return_value = sess
        yield sess


def test_parses_rows_in_search_order(session):
    """Rows follow esearch order, titles keep inline markup, labels lead sections."""
    session.get.return_value = _search("222", "111", "333")
    session.post.return_value = _resp(content=_xml(ARTICLE, BOOK, LETTER))

    rows = PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="lung", n=3)

    assert [r["id"] for r in rows] == ["222", "111", "333"]
    letter, article, book = rows
    assert article["title"] == "Solitary multicystic lung cancer: two case reports."
    assert article["summary"] == (
        "BACKGROUND: Rare disease.\n\nCASE PRESENTATION: Two elderly men."
    )
    assert article["text"].startswith(article["title"] + "\n\nBACKGROUND:")
    assert article["authors"] == ["Tang X", "Lung Study Group"]
    assert article["publication_types"] == ["Case Reports", "Journal Article"]
    assert article["mesh_terms"] == ["Lung Neoplasms"]
    assert (article["journal"], article["year"]) == ("BMC pulmonary medicine", "2021")
    assert (article["doi"], article["pmcid"]) == ("10.1186/x", "PMC8591847")
    assert article["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert letter["summary"] == "" and letter["text"] == letter["title"]
    assert letter["year"] == "2024"
    assert (book["journal"], book["summary"]) == (
        "StatPearls",
        "Overview of lung cancer.",
    )


def test_pmids_pubmed_omits_are_dropped(session):
    """A PMID that efetch does not return gives no row."""
    session.get.return_value = _search("111", "999")
    session.post.return_value = _resp(content=_xml(ARTICLE))
    rows = PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="lung", n=2)
    assert [r["id"] for r in rows] == ["111"]


def test_search_params_and_fetch_batches(session):
    """The search asks for n by relevance; efetch POSTs 200 PMIDs at a time."""
    pmids = [str(i) for i in range(450)]
    session.get.return_value = _search(*pmids)
    session.post.return_value = _resp(content=_xml())

    PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="lung", n=450)

    params = session.get.call_args.kwargs["params"]
    assert (params["term"], params["retmax"], params["sort"]) == (
        "lung",
        450,
        "relevance",
    )
    assert params["tool"] == "pyforger" and "api_key" not in params
    batches = [
        c.kwargs["data"]["id"].count(",") + 1 for c in session.post.call_args_list
    ]
    assert batches == [200, 200, 50]


def test_sort_override(session):
    """``sort=`` reaches esearch."""
    session.get.return_value = _search()
    PubMedRetrievalSource(sort="pub_date").fetch(RetrieveMode.QUERY, query="x", n=1)
    assert session.get.call_args.kwargs["params"]["sort"] == "pub_date"


def test_api_key_from_env_is_sent_and_kept_out_of_cache_key(session, monkeypatch):
    """NCBI_API_KEY is sent, but two keys share one cache key."""
    monkeypatch.setenv("NCBI_API_KEY", "secret-key")
    session.get.return_value = _search()
    source = PubMedRetrievalSource()
    source.fetch(RetrieveMode.QUERY, query="x", n=1)
    assert session.get.call_args.kwargs["params"]["api_key"] == "secret-key"
    assert "secret-key" not in repr(source.cache_key())
    assert source.cache_key() == PubMedRetrievalSource(api_key="other").cache_key()


def test_spacing_without_and_with_key(no_sleep):
    """Keyless requests are spaced 0.4 s apart, keyed ones 0.125 s."""
    limiter = pubmed_mod._RateLimiter()
    with patch.object(pubmed_mod.time, "monotonic", return_value=100.0):
        for _ in range(3):
            limiter.wait(0.4)
    assert no_sleep == pytest.approx([0.4, 0.8])
    assert PubMedRetrievalSource._INTERVAL_NO_KEY >= 1 / 3
    assert PubMedRetrievalSource._INTERVAL_KEY >= 1 / 10


def test_429_pauses_every_thread():
    """After a 429 the shared limiter holds the next request of any thread."""
    limiter = pubmed_mod._RateLimiter()
    with patch.object(pubmed_mod.time, "monotonic", return_value=50.0):
        limiter.pause(5.0)
        assert limiter._next == 55.0


def test_429_is_retried_after_retry_after(session, no_sleep):
    """A 429 waits at least Retry-After seconds, then succeeds."""
    session.get.side_effect = [
        _resp(429, headers={"Retry-After": "3"}),
        _search("111"),
    ]
    session.post.return_value = _resp(content=_xml(ARTICLE))
    rows = PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="x", n=1)
    assert [r["id"] for r in rows] == ["111"]
    assert 3.0 in no_sleep


def test_server_errors_retry_then_raise(session, no_sleep, monkeypatch):
    """5xx responses retry 5 times (1-16 s apart), then raise."""
    monkeypatch.setattr(pubmed_mod._LIMITER, "wait", lambda interval: None)
    session.get.return_value = _resp(503)
    with pytest.raises(requests.HTTPError):
        PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="x", n=1)
    assert session.get.call_count == PubMedRetrievalSource._MAX_RETRIES + 1
    assert no_sleep == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_connection_error_is_retried(session):
    """A dropped connection is retried."""
    session.get.side_effect = [requests.ConnectionError("reset"), _search()]
    assert PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="x", n=1) == []
    assert session.get.call_count == 2


def test_client_error_is_not_retried(session):
    """A 400 raises at once."""
    session.get.return_value = _resp(400)
    with pytest.raises(requests.HTTPError):
        PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="x", n=1)
    assert session.get.call_count == 1


def test_search_error_body_raises(session):
    """An esearch error body raises with its message."""
    session.get.return_value = _resp(json={"error": "Invalid query"})
    with pytest.raises(RuntimeError, match="Invalid query"):
        PubMedRetrievalSource().fetch(RetrieveMode.QUERY, query="x", n=1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mode": RetrieveMode.RANDOM, "query": "x"}, "QUERY mode only"),
        ({"mode": RetrieveMode.QUERY, "query": None}, "needs a query"),
        ({"mode": RetrieveMode.QUERY, "query": "x", "n": 10_001}, "10000"),
    ],
)
def test_invalid_calls(kwargs, match):
    """RANDOM mode, no query and n over 10,000 raise ValueError."""
    with pytest.raises(ValueError, match=match):
        PubMedRetrievalSource().fetch(**kwargs)


def test_retrieve_from_pubmed_stamps_queries_and_ignores_key_in_cache(
    tmp_path, monkeypatch
):
    """Each row gets its query; a cached call hits again with a different key."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))
    calls: list[str | None] = []

    def fake_fetch(self, mode, *, query=None, n=10, seed=None, **kw):
        calls.append(query)
        return [{"text": f"t {query}", "id": "1", "title": "t"}]

    with patch.object(PubMedRetrievalSource, "fetch", fake_fetch):
        rows = retrieve_from_pubmed(["a", "b"], n=1, api_key="k1")
        again = retrieve_from_pubmed(["a", "b"], n=1, api_key="k2")
    assert [r["query"] for r in rows] == ["a", "b"]
    assert again == rows and calls == ["a", "b"]


@pytest.mark.skipif(
    not os.environ.get("PYFORGER_PUBMED_LIVE"),
    reason="Set PYFORGER_PUBMED_LIVE=1 to run the live PubMed test",
)
def test_live_case_reports():
    """Live: a plain query returns Case Reports records."""
    rows = PubMedRetrievalSource().fetch(
        RetrieveMode.QUERY, query="lung cancer case reports", n=5
    )
    assert len(rows) == 5
    assert all("Case Reports" in r["publication_types"] for r in rows)
