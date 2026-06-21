"""Tests for the REPL's pure routing/parsing helpers."""

from __future__ import annotations

from pi_py_agent.app import classify_turn_input, parse_approval
from pi_py_sdk import ExtensionUIRequest


def _req(method: str, **kw) -> ExtensionUIRequest:
    return ExtensionUIRequest(id="x", method=method, **kw)


def test_classify_steer_default():
    assert classify_turn_input("focus on the parser") == ("steer", "focus on the parser")


def test_classify_follow_up_prefix():
    assert classify_turn_input("+also add tests") == ("follow_up", "also add tests")


def test_classify_abort():
    assert classify_turn_input("/abort") == ("abort", "")
    assert classify_turn_input("/stop") == ("abort", "")


def test_classify_exit_during_turn():
    assert classify_turn_input("/exit") == ("exit", "")
    assert classify_turn_input("/quit") == ("exit", "")


def test_parse_confirm_yes_no():
    req = _req("confirm", title="Allow?")
    assert parse_approval(req, "y") is True
    assert parse_approval(req, "yes") is True
    assert parse_approval(req, "") is False
    assert parse_approval(req, "n") is False


def test_parse_select_valid_and_invalid():
    req = _req("select", title="Pick", options=["A", "B", "C"])
    assert parse_approval(req, "2") == "C"
    assert parse_approval(req, "9") is None  # out of range
    assert parse_approval(req, "x") is None  # not a number


def test_parse_input_value_or_cancel():
    req = _req("input", title="Name")
    assert parse_approval(req, "Ada") == "Ada"
    assert parse_approval(req, "") is None
