"""Framing tests — the part that must match Pi's jsonl.ts byte-for-byte.

These have no Node/pydantic-runtime dependency beyond importing the package.
"""

from __future__ import annotations

import json

from pi_py_sdk.jsonl import JsonlDecoder, serialize_line


def test_serialize_line_appends_single_lf():
    out = serialize_line({"type": "prompt", "message": "hi"})
    assert out.endswith(b"\n")
    assert out.count(b"\n") == 1
    assert json.loads(out.decode("utf-8")) == {"type": "prompt", "message": "hi"}


def test_single_line():
    dec = JsonlDecoder()
    assert dec.feed(b'{"a":1}\n') == ['{"a":1}']


def test_multiple_lines_one_chunk():
    dec = JsonlDecoder()
    assert dec.feed(b"a\nb\nc\n") == ["a", "b", "c"]


def test_partial_line_across_chunks():
    dec = JsonlDecoder()
    assert dec.feed(b'{"a":') == []
    assert dec.feed(b"1}\n") == ['{"a":1}']


def test_trailing_cr_is_stripped():
    dec = JsonlDecoder()
    assert dec.feed(b"hello\r\n") == ["hello"]


def test_lone_cr_not_at_eol_is_preserved():
    dec = JsonlDecoder()
    # \r in the middle must survive; only a single *trailing* \r is stripped.
    assert dec.feed(b"a\rb\n") == ["a\rb"]


def test_unicode_line_separator_inside_string_is_not_a_delimiter():
    # U+2028 (LINE SEPARATOR) is valid inside a JSON string. We must NOT split on it.
    payload = {"text": "before after"}
    line = json.dumps(payload, ensure_ascii=False)
    dec = JsonlDecoder()
    lines = dec.feed((line + "\n").encode("utf-8"))
    assert len(lines) == 1
    assert json.loads(lines[0]) == payload


def test_multibyte_char_split_across_chunks():
    # '€' is 3 UTF-8 bytes (E2 82 AC); split it mid-character across feeds.
    data = '{"c":"€"}\n'.encode("utf-8")
    split = data[:5], data[5:]
    dec = JsonlDecoder()
    assert dec.feed(split[0]) == []
    out = dec.feed(split[1])
    assert json.loads(out[0]) == {"c": "€"}


def test_flush_emits_nonempty_remainder():
    dec = JsonlDecoder()
    assert dec.feed(b'{"a":1}') == []
    assert dec.flush() == ['{"a":1}']


def test_flush_ignores_empty_remainder():
    dec = JsonlDecoder()
    assert dec.feed(b"x\n") == ["x"]
    assert dec.flush() == []


def test_blank_lines_are_emitted_as_empty_strings():
    dec = JsonlDecoder()
    assert dec.feed(b"\n\n") == ["", ""]
