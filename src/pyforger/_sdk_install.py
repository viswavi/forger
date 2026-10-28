"""Exact SDK sources shared by host-independent runtime installers."""

from __future__ import annotations

import shlex

SDK_VERSION = "1.58.0"
SDK_COMMIT = "82327f93b4d02518a08c720f0111fa4aa13fce41"
SDK_REQUIREMENT = (
    "openhands-sdk @ git+https://github.com/neubig/software-agent-sdk.git"
    f"@{SDK_COMMIT}#subdirectory=openhands-sdk"
)
TOOLS_REQUIREMENT = SDK_REQUIREMENT.replace("openhands-sdk", "openhands-tools")


def sdk_provenance_command(python: str = "/opt/openhands-sdk-venv/bin/python") -> str:
    script = f"""import importlib, importlib.metadata as m, json
for requirement in {(SDK_REQUIREMENT, TOOLS_REQUIREMENT)!r}:
    name, source = requirement.split(' @ git+', 1)
    url, revision = source.split('@', 1)
    commit = revision.split('#', 1)[0]
    direct = json.loads(m.distribution(name).read_text('direct_url.json') or '{{}}')
    assert m.version(name) == {SDK_VERSION!r}, (name, 'unexpected SDK version')
    assert direct.get('subdirectory') == name, (name, 'unexpected SDK subdirectory')
    assert direct.get('url') == url, (name, 'unexpected SDK repository')
    assert direct.get('vcs_info', {{}}).get('vcs') == 'git', (name, 'unexpected SDK source type')
    assert direct.get('vcs_info', {{}}).get('commit_id') == commit, (name, 'unexpected SDK commit')
    importlib.import_module(name.replace('-', '.'))
"""
    return shlex.quote(python) + " -c " + shlex.quote(f"exec({script!r})")
