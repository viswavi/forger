"""Extraction of problem/solution/test rows from GitHub repos, PDFs, and webpages.

Each ``extract_problems_from_*`` function returns a list of rows shaped like:

    {
        "source": "github" | "pdf" | "webpage",
        "source_url": str,
        "source_path": str | None,   # path within a repo/PDF, if applicable
        "title": str,
        "problem_statement": str,
        "solutions": [{"language": str, "code": str, "path": str}],
        "tests": [{"input": str, "output": str}],
        "metadata": dict,
    }

Extraction is purely heuristic (regex/filename/HTML-structure matching) --
no LLM calls. Heuristics are generic best-effort defaults; a given source
will often need a bit of source-specific tuning (see the ``codeforces.com``
handler in ``extract_problems_from_webpage`` as an example of that).
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from pypdf import PdfReader

from pyforger import op
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("extraction")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_CACHE_DIR = BASE_DIR / "repo_cache"
DEFAULT_PDF_CACHE_DIR = BASE_DIR / "pdf_cache"

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_REQUEST_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_REQUEST_TIMEOUT = 20
_GIT_CLONE_TIMEOUT = 120

# github.com path segments that are site features, not usernames/orgs -- a URL
# like github.com/topics/icpc is a topic-listing page, not a repo to clone.
_GITHUB_RESERVED_PATHS = {
    "about", "account", "apps", "articles", "blog", "business", "codespaces",
    "collections", "contact", "customer-stories", "dashboard", "discussions",
    "education", "enterprise", "events", "explore", "features", "gist",
    "home", "issues", "join", "login", "logout", "marketplace", "new",
    "notifications", "org", "orgs", "pricing", "pulls", "readme", "search",
    "security", "settings", "site", "sponsors", "stars", "topics",
    "trending", "watching", "support", "resources", "solutions", "team",
    "teams", "sitemap",
}
_GITHUB_USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")

# -- shared helpers -----------------------------------------------------------

_SOLUTION_EXTS = {
    ".py": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".kt": "kotlin",
    ".js": "javascript",
}
_SOLUTION_NAME_HINTS = ("accepted", "solution", "sol", "answer", "ac_", "main")
_STATEMENT_EXTS = {".md", ".txt", ".rst", ".pdf", ".html"}
_STATEMENT_NAME_HINTS = ("statement", "problem", "description", "readme")
_TEST_DIR_NAMES = ("tests", "test", "data", "samples", "sample")

# Matches "Sample Input ... Sample Output ..." blocks inside a plain-text
# problem statement -- the convention nearly every ICPC/IOI-style judge uses.
_SAMPLE_IO_RE = re.compile(
    r"(?:sample|example)\s+input[^\n]*\n(.*?)\n(?:sample|example)\s+output[^\n]*\n(.*?)"
    r"(?=\n(?:sample|example)\s+input|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_sample_io(text: str) -> list[dict]:
    """Best-effort 'Sample Input' / 'Sample Output' block pairing from plain text."""
    tests = []
    for m in _SAMPLE_IO_RE.finditer(text):
        inp, out = m.group(1).strip(), m.group(2).strip()
        if inp and out:
            tests.append({"input": inp, "output": out})
    return tests


def write_jsonl(rows: list[dict], path: str | os.PathLike) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# -- GitHub extraction ---------------------------------------------------------


def _repo_slug(repo: str) -> str:
    """Normalize 'owner/name', a full github.com URL, or a .git URL to 'owner/name'."""
    if repo.startswith("http://") or repo.startswith("https://") or repo.startswith("git@"):
        parsed = urlparse(repo if "://" in repo else f"ssh://{repo}")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"Could not parse owner/repo from URL: {repo}")
        owner, name = parts[0], parts[1]
        return f"{owner}/{name.removesuffix('.git')}"
    return repo.removesuffix(".git")


def _repo_url(repo: str) -> str:
    slug = _repo_slug(repo)
    return f"https://github.com/{slug}"


def _clone_repo(repo: str, cache_dir: Path) -> Path:
    """Shallow-clone repo into cache_dir if not already present; return its path."""
    slug = _repo_slug(repo)
    dest = cache_dir / slug.replace("/", "__")
    if dest.exists():
        logger.info("Using existing clone of %s at %s", slug, dest)
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _repo_url(repo)
    logger.info("Cloning %s into %s", url, dest)
    # GIT_TERMINAL_PROMPT=0 makes git fail fast (instead of blocking on a
    # username/password prompt) when the repo is private, renamed, or
    # doesn't exist -- e.g. a misclassified non-repo github.com URL.
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
        timeout=_GIT_CLONE_TIMEOUT,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return dest


def _read_text_or_pdf(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _pdf_to_text(path)
    return path.read_text(errors="replace")


def _pick_statement_file(files: list[Path]) -> Path | None:
    candidates = [f for f in files if f.suffix.lower() in _STATEMENT_EXTS]

    def score(f: Path) -> int:
        name = f.stem.lower()
        for i, hint in enumerate(_STATEMENT_NAME_HINTS):
            if hint in name:
                return i
        return len(_STATEMENT_NAME_HINTS)

    candidates.sort(key=score)
    return candidates[0] if candidates else None


def _pick_solution_files(files: list[Path], max_solutions: int = 3) -> list[Path]:
    candidates = [f for f in files if f.suffix.lower() in _SOLUTION_EXTS]

    def score(f: Path) -> int:
        name = f.stem.lower()
        for i, hint in enumerate(_SOLUTION_NAME_HINTS):
            if hint in name:
                return i
        return len(_SOLUTION_NAME_HINTS)

    candidates.sort(key=score)
    return candidates[:max_solutions]


def _pair_tests(files: list[Path]) -> list[dict]:
    """Pair up sample-test files by shared basename or an input/output naming convention."""
    in_by_key: dict[str, Path] = {}
    out_by_key: dict[str, Path] = {}
    io_name_re = re.compile(
        r"^(?:sample_?|test_?)?(?P<kind>input|output|in|out|ans|expected)[-_.]?(?P<idx>\d*)$",
        re.IGNORECASE,
    )
    for f in files:
        ext = f.suffix.lower()
        stem = f.stem.lower()
        if ext == ".in":
            in_by_key[stem] = f
            continue
        if ext in (".out", ".ans"):
            out_by_key[stem] = f
            continue
        m = io_name_re.match(stem)
        if not m:
            continue
        kind, idx = m.group("kind").lower(), m.group("idx") or "0"
        if kind in ("input", "in"):
            in_by_key[idx] = f
        else:
            out_by_key[idx] = f

    tests = []
    for key in sorted(set(in_by_key) & set(out_by_key)):
        try:
            tests.append(
                {
                    "input": in_by_key[key].read_text(errors="replace"),
                    "output": out_by_key[key].read_text(errors="replace"),
                }
            )
        except OSError as e:
            logger.warning("Skipping unreadable test pair for key %r: %s", key, e)
    return tests


@op
def extract_problems_from_github(
    repo: str,
    *,
    max_problems: int | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> list[dict]:
    """Clone repo and heuristically pull out (statement, solution, tests) triples.

    A directory is treated as a "problem" when it directly contains both a
    statement-like file (readme/statement/problem, .md/.txt/.rst/.pdf/.html)
    and at least one solution-like source file. Sample tests are pulled from
    files directly in that directory plus any tests/test/data/samples
    subdirectory, paired by matching basename (foo.in/foo.out) or an
    input/output naming convention (input1.txt/output1.txt, in2/out2, ...).

    This default heuristic works well for repos laid out as one directory per
    problem; repos with other conventions (e.g. a single top-level README
    listing many problems) will need a source-specific pass instead.
    """
    root = _clone_repo(repo, Path(cache_dir) if cache_dir else DEFAULT_REPO_CACHE_DIR)
    problems: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        dpath = Path(dirpath)
        files = [dpath / fn for fn in filenames]

        statement_file = _pick_statement_file(files)
        solution_files = _pick_solution_files(files)
        if statement_file is None or not solution_files:
            continue

        try:
            statement_text = _read_text_or_pdf(statement_file)
        except OSError as e:
            logger.warning("Could not read statement %s: %s", statement_file, e)
            continue

        solutions = []
        for f in solution_files:
            try:
                code = f.read_text(errors="replace")
            except OSError as e:
                logger.warning("Could not read solution %s: %s", f, e)
                continue
            solutions.append(
                {
                    "language": _SOLUTION_EXTS[f.suffix.lower()],
                    "code": code,
                    "path": str(f.relative_to(root)),
                }
            )

        test_candidate_files = list(files)
        for sub in _TEST_DIR_NAMES:
            subdir = dpath / sub
            if subdir.is_dir():
                test_candidate_files.extend(p for p in subdir.rglob("*") if p.is_file())
        tests = _pair_tests(test_candidate_files)
        if not tests:
            tests = _extract_sample_io(statement_text)

        problems.append(
            {
                "source": "github",
                "source_url": _repo_url(repo),
                "source_path": str(dpath.relative_to(root)),
                "title": statement_file.stem,
                "problem_statement": statement_text,
                "solutions": solutions,
                "tests": tests,
                "metadata": {"repo": _repo_slug(repo)},
            }
        )
        if max_problems and len(problems) >= max_problems:
            break
    return problems


# -- PDF extraction -------------------------------------------------------------


def _pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _download_pdf(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".pdf")
    if dest.exists():
        return dest
    resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


@op
def extract_problems_from_pdf(
    pdf_source: str,
    *,
    split_regex: str | None = None,
    title: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> list[dict]:
    """Extract text from a PDF (URL or local path) and split it into problem rows.

    Sample-test blocks ("Sample Input" / "Sample Output") inside the extracted
    text are pulled out heuristically; solutions are never present in a PDF
    statement so ``solutions`` is always empty here.

    A single contest PDF often bundles multiple problems. Splitting them
    apart reliably requires a pattern specific to that booklet's layout
    (page headers, "Problem X" markers, etc.) -- pass ``split_regex``, a
    regex with one capturing group for the problem title, matched against
    each problem's leading line; everything up to the next match becomes
    that problem's body. Without ``split_regex`` the whole PDF is returned
    as a single row.
    """
    is_url = pdf_source.startswith("http://") or pdf_source.startswith("https://")
    path = (
        _download_pdf(pdf_source, Path(cache_dir) if cache_dir else DEFAULT_PDF_CACHE_DIR)
        if is_url
        else Path(pdf_source)
    )
    text = _pdf_to_text(path)
    default_title = title or Path(urlparse(pdf_source).path if is_url else pdf_source).stem

    if not split_regex:
        segments = [(default_title, text)]
    else:
        pattern = re.compile(split_regex, re.MULTILINE)
        matches = list(pattern.finditer(text))
        if not matches:
            segments = [(default_title, text)]
        else:
            segments = []
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                segments.append((m.group(1).strip(), text[start:end].strip()))

    rows = []
    for seg_title, body in segments:
        rows.append(
            {
                "source": "pdf",
                "source_url": pdf_source,
                "source_path": None,
                "title": seg_title,
                "problem_statement": body,
                "solutions": [],
                "tests": _extract_sample_io(body),
                "metadata": {},
            }
        )
    return rows


# -- Webpage extraction ----------------------------------------------------------


def _preceding_label(tag: Tag) -> str:
    node = tag.find_previous(["h1", "h2", "h3", "h4", "b", "strong", "div", "span", "p"])
    return node.get_text(strip=True) if node is not None else ""


def _extract_pre_pairs_from_html(html: str) -> list[dict]:
    """Pair up consecutive <pre> blocks whose nearby text mentions input/output.

    Matches the common judge-page convention of an "Input"-labeled <pre>
    immediately followed by an "Output"-labeled <pre>.
    """
    soup = BeautifulSoup(html, "html.parser")
    pres = soup.find_all("pre")
    tests = []
    i = 0
    while i < len(pres) - 1:
        label_a = _preceding_label(pres[i]).lower()
        label_b = _preceding_label(pres[i + 1]).lower()
        if "input" in label_a and "output" in label_b:
            tests.append(
                {
                    "input": pres[i].get_text("\n").strip(),
                    "output": pres[i + 1].get_text("\n").strip(),
                }
            )
            i += 2
        else:
            i += 1
    return tests


def _extract_codeforces(html: str, url: str) -> dict | None:
    """Codeforces-specific handler: the problem-statement HTML block has a fixed layout."""
    soup = BeautifulSoup(html, "html.parser")
    statement = soup.find("div", class_="problem-statement")
    if statement is None:
        return None

    header = statement.find("div", class_="header")
    title_tag = header.find("div", class_="title") if header else None
    title = title_tag.get_text(strip=True) if title_tag else ""

    tests = []
    for sample in statement.find_all("div", class_="sample-test"):
        inputs = sample.find_all("div", class_="input")
        outputs = sample.find_all("div", class_="output")
        for inp, out in zip(inputs, outputs):
            inp_pre = inp.find("pre")
            out_pre = out.find("pre")
            if inp_pre is not None and out_pre is not None:
                tests.append(
                    {
                        "input": inp_pre.get_text("\n").strip(),
                        "output": out_pre.get_text("\n").strip(),
                    }
                )

    # Statement body text, excluding the sample-test blocks already captured above.
    body_parts = []
    for child in statement.find_all(["div", "p"], recursive=False):
        classes = child.get("class") or []
        if "sample-tests" in classes or "header" in classes:
            continue
        text = child.get_text("\n", strip=True)
        if text:
            body_parts.append(text)

    return {
        "source": "webpage",
        "source_url": url,
        "source_path": None,
        "title": title,
        "problem_statement": "\n\n".join(body_parts),
        "solutions": [],
        "tests": tests,
        "metadata": {"platform": "codeforces"},
    }


_SITE_HANDLERS = {
    "codeforces.com": _extract_codeforces,
}


@op
def extract_problems_from_webpage(url: str) -> list[dict]:
    """Fetch a problem page and extract (statement, tests) from it.

    Tries a site-specific handler first (currently just Codeforces, whose
    problem pages have a fixed, well-known HTML structure); otherwise falls
    back to a generic extraction: trafilatura pulls the main article text
    (stripping nav/boilerplate) for the statement, and sample tests are
    pulled either from paired <input>/<output>-labeled <pre> blocks in the
    raw HTML or, failing that, from "Sample Input"/"Sample Output" text
    blocks in the extracted statement. Other judges (AtCoder, SPOJ, ICPC
    archive mirrors, ...) have their own DOM quirks and will do better with
    a handler added to ``_SITE_HANDLERS`` the way Codeforces has one.
    """
    resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    html = resp.text

    netloc = urlparse(url).netloc.removeprefix("www.")
    handler = _SITE_HANDLERS.get(netloc)
    if handler is not None:
        row = handler(html, url)
        if row is not None:
            return [row]

    import trafilatura

    extracted_json = trafilatura.extract(
        html, output_format="json", with_metadata=True, url=url
    )
    if extracted_json:
        import json

        extracted = json.loads(extracted_json)
        title = extracted.get("title") or ""
        statement_text = extracted.get("text") or ""
    else:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        statement_text = soup.get_text("\n", strip=True)

    tests = _extract_pre_pairs_from_html(html) or _extract_sample_io(statement_text)

    return [
        {
            "source": "webpage",
            "source_url": url,
            "source_path": None,
            "title": title,
            "problem_statement": statement_text,
            "solutions": [],
            "tests": tests,
            "metadata": {"platform": netloc},
        }
    ]


# -- dispatch over a batch of search-result rows ---------------------------------


def classify_source(url: str) -> tuple[str, str]:
    """Return (kind, identifier) for a URL: 'github' + owner/repo, 'pdf' + url, or 'webpage' + url."""
    parsed = urlparse(url)
    netloc = parsed.netloc.removeprefix("www.")
    if netloc == "github.com":
        parts = [p for p in parsed.path.split("/") if p]
        if (
            len(parts) >= 2
            and parts[0].lower() not in _GITHUB_RESERVED_PATHS
            and _GITHUB_USERNAME_RE.match(parts[0])
        ):
            return "github", f"{parts[0]}/{parts[1]}"
    if parsed.path.lower().endswith(".pdf"):
        return "pdf", url
    return "webpage", url


def extract_problems_from_search_results(search_results: list[dict]) -> list[dict]:
    """Dispatch each search-result row's URL to the right extractor; skip failures."""
    seen_urls: set[str] = set()
    rows: list[dict] = []
    for result in search_results:
        url = result.get("url") or result.get("id")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        kind, identifier = classify_source(url)
        try:
            if kind == "github":
                extracted = extract_problems_from_github(identifier)
            elif kind == "pdf":
                extracted = extract_problems_from_pdf(identifier)
            else:
                extracted = extract_problems_from_webpage(identifier)
        except Exception as e:
            logger.warning("Extraction failed for %s (%s): %s", url, kind, e)
            continue
        rows.extend(extracted)
    return rows
