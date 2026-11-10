import socket

import pytest
import pytest_socket

from krta.redact import Redactor


def test_redacts_env_secret_values():
    r = Redactor.from_env({"DART_API_KEY": "abcd1234", "KRX_PW": "pw-9999", "HOME": "/home/x"})
    out = r.redact("key=abcd1234 pw=pw-9999 home=/home/x")
    assert "abcd1234" not in out and "pw-9999" not in out and "/home/x" in out


def test_redacts_url_key_params_even_if_unknown():
    r = Redactor([])
    assert r.redact("https://opendart.fss.or.kr/api/list.json?crtfc_key=ZZZ&corp_code=1") \
        == "https://opendart.fss.or.kr/api/list.json?crtfc_key=***&corp_code=1"


def test_short_values_not_treated_as_secret():
    r = Redactor.from_env({"KRX_ID": "ab"})
    assert r.redact("ab cd") == "ab cd"


def test_redacts_all_url_key_param_names():
    r = Redactor([])
    out = r.redact("https://x.example/a?AUTH_KEY=AAA1&api_key=BBB2&key=CCC3&page=2")
    assert out == "https://x.example/a?AUTH_KEY=***&api_key=***&key=***&page=2"


def test_url_param_suffix_names_not_masked():
    # monkey=... must not be caught by the "key" rule
    r = Redactor([])
    assert r.redact("https://x.example/a?monkey=banana") == "https://x.example/a?monkey=banana"


def test_longer_secret_replaced_first():
    # "abcd" is a prefix of "abcd5678"; replacing the short one first would leak "5678"
    r = Redactor(["abcd", "abcd5678"])
    out = r.redact("v=abcd5678")
    assert "5678" not in out and "abcd" not in out


def test_env_name_pattern():
    env = {
        "ANTHROPIC_API_KEY": "dummy-anthropic",
        "KRX_API_KEY": "dummy-krx",
        "KRX_ID": "dummy-id",
        "SOME_TOKEN": "dummy-token",
        "DB_PASSWORD": "dummy-password",
        "MY_SECRET": "dummy-secret",
        "PATH": "/usr/bin",
        "KEYBOARD": "us-layout",
    }
    out = Redactor.from_env(env).redact(" ".join(env.values()))
    for name in ("ANTHROPIC_API_KEY", "KRX_API_KEY", "KRX_ID", "SOME_TOKEN", "DB_PASSWORD", "MY_SECRET"):
        assert env[name] not in out
    assert "/usr/bin" in out and "us-layout" in out


def test_regex_special_chars_in_secret():
    r = Redactor(["a.b*c(d)"])
    assert r.redact("x a.b*c(d) y aXbbc(d)") == "x *** y aXbbc(d)"


def test_empty_text_and_no_secrets():
    assert Redactor([]).redact("") == ""
    assert Redactor([]).redact("plain text") == "plain text"


def test_percent_encoded_secret_masked():
    # a password with special chars appears percent-encoded in form bodies / URLs
    r = Redactor(["p@ss w&rd"])
    out = r.redact("pw=p%40ss%20w%26rd&x=1 pw2=p%40ss+w%26rd raw=p@ss w&rd")
    assert "p%40ss" not in out and "p@ss" not in out


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
def test_network_is_disabled_in_tests():
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.socket()


def test_add_registers_runtime_secret():
    r = Redactor(["abcd5678"])
    assert r.redact("tok-runtime-1 abcd5678") == "tok-runtime-1 ***"
    r.add("tok-runtime-1")
    assert r.redact("tok-runtime-1 abcd5678") == "*** ***"
    # Percent-encoded forms are masked too, as for constructor secrets.
    r.add("a/b+c=d")
    assert r.redact("x=a%2Fb%2Bc%3Dd") == "x=***"


def test_add_ignores_short_or_empty_values():
    r = Redactor([])
    r.add("")
    r.add("abc")
    assert r.redact("abc def") == "abc def"
