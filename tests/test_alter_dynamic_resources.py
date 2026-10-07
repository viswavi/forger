"""Tests for dynamic pipeline operations."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyforger.alter import _dynamic_resources as resources


def test_raw_text_is_unchanged():
    """Test raw text is unchanged."""
    assert resources.prepare_input("plain input") == "plain input"


def test_html_url_is_extracted(monkeypatch):
    """Test html url is extracted."""
    monkeypatch.setattr(
        resources,
        "_download",
        lambda url: (
            b"<html><style>hidden</style><body>Hello <b>world</b></body></html>",
            "text/html",
        ),
    )
    result = json.loads(resources.prepare_input("https://example.org/page"))
    assert result["kind"] == "html"
    assert "Hello" in result["content"]
    assert "hidden" not in result["content"]


def test_pdf_url_is_parsed(monkeypatch):
    # A small valid PDF with an actual text drawing operation. A blank PDF
    # would check parsing but would not exercise PdfReader.extract_text().
    """Test pdf url is parsed."""
    message = b"BT /F1 12 Tf 20 100 Td (Hello from PDF) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(message)).encode()
        + b" >>\nstream\n"
        + message
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF"
    )
    monkeypatch.setattr(
        resources,
        "_download",
        lambda url: (
            bytes(pdf),
            "application/pdf",
        ),
    )
    result = json.loads(resources.prepare_input("https://example.org/file.pdf"))
    assert result["kind"] == "pdf"
    assert result["content"].strip() == "Hello from PDF"


def test_git_url_is_cloned_to_bounded_text_bundle(monkeypatch):
    """Test git url is cloned to bounded text bundle."""
    monkeypatch.setattr(resources, "_public_https", lambda url: "github.com")
    monkeypatch.setenv("LITELLM_API_KEY", "dummy")

    def fake_run(command, *, cwd, env, capture_output, text, timeout):
        assert "LITELLM_API_KEY" not in env
        repo = Path(command[-1])
        repo.mkdir()
        (repo / "main.py").write_text("def parse(): return 42")
        (repo / "binary.bin").write_bytes(b"\0binary")
        (repo / ".git").mkdir()
        (repo / ".git" / "config").write_text("internal")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(resources.subprocess, "run", fake_run)
    result = json.loads(resources.prepare_input("https://github.com/acme/repo.git"))
    assert result["kind"] == "git"
    assert result["files"] == {"main.py": "def parse(): return 42"}


def test_git_rejects_other_hosts(monkeypatch):
    """Test git rejects other hosts."""
    monkeypatch.setattr(resources, "_public_https", lambda url: "example.org")
    with pytest.raises(ValueError, match="GitHub or GitLab"):
        resources.prepare_input("https://example.org/repo.git")


def test_https_rejects_private_addresses(monkeypatch):
    """Test https rejects private addresses."""
    monkeypatch.setattr(
        resources.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="non-public"):
        resources._public_https("https://localhost/document.pdf")


def test_download_validates_redirect_target(monkeypatch):
    """Test download validates redirect target."""
    monkeypatch.setattr(
        resources,
        "_public_https",
        lambda url: (
            (_ for _ in ()).throw(ValueError("blocked"))
            if "internal" in url
            else "example.org"
        ),
    )

    class Response:
        status_code = 302
        headers = {"location": "http://internal/file"}

        def close(self):
            """Close the mocked response."""

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="blocked"):
        resources._download("https://example.org/start")


def test_git_tool_lists_and_reads_prepared_files(monkeypatch):
    """Test git tool lists and reads prepared files."""
    from pyforger.alter import _dynamic_tool as tool

    prepared = json.dumps(
        {
            "kind": "git",
            "url": "https://github.com/a/b.git",
            "files": {"README.md": "hello", "src/a.py": "x = 1"},
        }
    )
    monkeypatch.setattr(tool, "prepare_input", lambda item: prepared)
    session = tool.PythonExecutor("https://github.com/a/b.git", 5)
    git = tool.GitExecutor(session)
    listing = git(tool.GitAction(operation="list"))
    assert json.loads(listing.result_text)["files"] == ["README.md", "src/a.py"]
    content = git(tool.GitAction(operation="read", path="src/a.py"))
    assert json.loads(content.result_text)["content"] == "x = 1"
    assert session.prepared_input() == prepared


def test_sdk_resolves_all_restricted_tools():
    """Test sdk resolves all restricted tools."""
    from openhands.sdk.tool import Tool
    from openhands.sdk.tool.registry import resolve_tool

    from pyforger.alter import _dynamic_tool as tool

    session = tool.PythonExecutor("plain input", 5)
    session_id = tool.register_executor(session)
    try:
        for name in ("ForgerPythonTool", "ForgerResourceTool", "ForgerGitTool"):
            resolved = resolve_tool(
                Tool(name=name, params={"session_id": session_id}), None
            )
            assert len(resolved) == 1
            assert resolved[0].executor is not None
    finally:
        tool.unregister_executor(session_id)


def test_generated_program_receives_prepared_url(monkeypatch):
    """Test generated program receives prepared url."""
    from pyforger.alter import dynamic

    prepared = json.dumps(
        {"kind": "html", "url": "https://example.org", "content": "hello"}
    )
    monkeypatch.setattr(dynamic, "prepare_input", lambda item: prepared)
    code = 'import json\ndef run(item):\n    return json.loads(item)["content"]'
    assert dynamic._execute(code, "https://example.org", 5) == "hello"
