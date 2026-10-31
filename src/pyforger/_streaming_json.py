"""Incremental JSON traversal, bounded by the largest scalar or selected value."""

from __future__ import annotations

import json
import re
from pathlib import Path


class JSONReader:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.position = 0

    def _char(self):
        if self.position == len(self.buffer):
            self.buffer = self.stream.read(65536)
            self.position = 0
        return self.buffer[self.position : self.position + 1]

    def token(self, whitespace_target=None):
        while self._char():
            match = re.match(r"[ \t\r\n]+", self.buffer[self.position :])
            if match is None:
                break
            if whitespace_target is not None:
                whitespace_target.write(match[0])
            self.position += len(match[0])
        prefix = ""
        char = self._char()
        if not char:
            return prefix, ""
        if char in "{}[]:,":
            self.position += 1
            return prefix, char
        parts = []
        if char == '"':
            parts.append(char)
            self.position += 1
            while self._char():
                match = re.search(r'["\\]', self.buffer[self.position :])
                if match is None:
                    parts.append(self.buffer[self.position :])
                    self.position = len(self.buffer)
                    continue
                end = self.position + match.start()
                parts.append(self.buffer[self.position : end + 1])
                self.position = end + 1
                char = self.buffer[end]
                if char == '"':
                    break
                else:
                    # Consume the escaped character even across a chunk boundary.
                    char = self._char()
                    if not char:
                        raise ValueError("Unterminated JSON escape")
                    parts.append(char)
                    self.position += 1
            else:
                raise ValueError("Unterminated JSON string")
        else:
            while char := self._char():
                if char in " \t\r\n{}[]:,":
                    break
                self.position += 1
                parts.append(char)
        raw = "".join(parts)
        json.loads(raw)
        return prefix, raw

    def value(self, token=None):
        _, raw = self.token() if token is None else token
        if raw == "{":
            result = {}
            token = self.token()
            if token[1] == "}":
                return result
            while True:
                key = json.loads(token[1])
                if not isinstance(key, str) or self.token()[1] != ":":
                    raise ValueError("Invalid JSON object")
                result[key] = self.value()
                token = self.token()
                if token[1] == "}":
                    return result
                if token[1] != ",":
                    raise ValueError("Invalid JSON object separator")
                token = self.token()
        if raw == "[":
            result = []
            token = self.token()
            if token[1] == "]":
                return result
            while True:
                result.append(self.value(token))
                token = self.token()
                if token[1] == "]":
                    return result
                if token[1] != ",":
                    raise ValueError("Invalid JSON array separator")
                token = self.token()
        return json.loads(raw)

    def rewrite(self, target, clean, sensitive, token=None):
        prefix, raw = self.token(target) if token is None else token
        target.write(prefix)
        if raw not in ("{", "["):
            value = json.loads(raw)
            redacted = clean(value)
            target.write(raw if redacted == value else json.dumps(redacted))
            return
        target.write(raw)
        closing = "}" if raw == "{" else "]"
        token = self.token(target)
        if token[1] == closing:
            target.write("".join(token))
            return
        while True:
            if raw == "{":
                key = json.loads(token[1])
                if not isinstance(key, str):
                    raise ValueError("Invalid JSON key")
                target.write("".join(token))
                colon = self.token(target)
                if colon[1] != ":":
                    raise ValueError("Invalid JSON object")
                target.write("".join(colon))
                token = self.token(target)
                if sensitive(key):
                    target.write(token[0] + '"[REDACTED]"')
                    self.rewrite(_Discard(), clean, sensitive, token)
                else:
                    self.rewrite(target, clean, sensitive, token)
            else:
                self.rewrite(target, clean, sensitive, token)
            token = self.token(target)
            if token[1] not in (",", closing):
                raise ValueError("Invalid JSON separator")
            target.write("".join(token))
            if token[1] == closing:
                return
            token = self.token(target)


class _Discard:
    def write(self, value):
        pass


def trajectory_events(path: Path):
    """Read native top-level steps/events arrays one complete event at a time."""
    with path.open(encoding="utf-8") as stream:
        reader = JSONReader(stream)
        token = reader.token()
        if token[1] == "{":
            token = reader.token()
            found = False
            while token[1] != "}":
                key = json.loads(token[1])
                if reader.token()[1] != ":":
                    raise ValueError("Invalid trajectory object")
                value = reader.token()
                if key in ("steps", "events"):
                    if found:
                        raise ValueError("Ambiguous trajectory arrays")
                    found = True
                    yield from _events(reader, value)
                else:
                    reader.rewrite(_Discard(), lambda v: v, lambda k: False, value)
                token = reader.token()
                if token[1] == "}":
                    break
                if token[1] != ",":
                    raise ValueError("Invalid trajectory object separator")
                token = reader.token()
        else:
            yield from _events(reader, token)
        if reader.token()[1]:
            raise ValueError("Trailing trajectory content")


def _events(reader, token):
    if token[1] != "[":
        raise ValueError("Harbor trajectory must contain a list of steps")
    token = reader.token()
    if token[1] == "]":
        return
    while True:
        yield reader.value(token)
        token = reader.token()
        if token[1] == "]":
            return
        if token[1] != ",":
            raise ValueError("Invalid trajectory array separator")
        token = reader.token()


def object_items(path: Path, array_keys=()):
    """Yield top-level values, expanding selected arrays one item at a time."""
    with path.open(encoding="utf-8") as stream:
        reader = JSONReader(stream)
        if reader.token()[1] != "{":
            raise ValueError("Expected a JSON object")
        token = reader.token()
        while token[1] != "}":
            key = json.loads(token[1])
            if not isinstance(key, str) or reader.token()[1] != ":":
                raise ValueError("Invalid JSON object")
            token = reader.token()
            if key in array_keys and token[1] == "[":
                for item in _events(reader, token):
                    yield key, item
            elif key in ("agent", "final_metrics"):
                yield key, reader.value(token)
            else:
                reader.rewrite(
                    _Discard(), lambda value: value, lambda key: False, token
                )
            token = reader.token()
            if token[1] == "}":
                break
            if token[1] != ",":
                raise ValueError("Invalid JSON object separator")
            token = reader.token()
        if reader.token()[1]:
            raise ValueError("Trailing JSON content")
