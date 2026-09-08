import pytest

from app.security import (
    ValidationError,
    model_name_from_url,
    parse_parameters,
    validate_download_url,
    validate_model_name,
)


@pytest.mark.parametrize("name", ["llama3", "my-model", "ns/model", "model:q4_k_m", "a.b_c-d"])
def test_valid_model_names(name):
    assert validate_model_name(name) == name.lower()


@pytest.mark.parametrize("name", ["", "../etc", "a b", "x" * 200, "-bad", "bad/", "a//b", ":tag"])
def test_invalid_model_names(name):
    with pytest.raises(ValidationError):
        validate_model_name(name)


def test_model_name_from_url():
    assert model_name_from_url("https://hf.co/r/Model.Q4_K_M.gguf") == "model.q4_k_m"
    assert model_name_from_url("https://example.com/path/") == "model"
    assert model_name_from_url("https://example.com/TheBloke--Thing.GGUF") == "thebloke-thing"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x.gguf",
        "file:///etc/passwd",
        "http://127.0.0.1/x.gguf",
        "http://localhost/x.gguf",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/x",
        "http://10.0.0.5/x.gguf",
        "http://192.168.1.10/x.gguf",
        "https://0.0.0.0/x",
    ],
)
def test_blocked_download_urls(url):
    with pytest.raises(ValidationError):
        validate_download_url(url, allow_private=False)


def test_private_allowed_when_configured():
    assert validate_download_url("http://10.0.0.5/x.gguf", allow_private=True)


def test_public_host_ip_literal_allowed():
    # 1.1.1.1 is a public address; no network call is made for an IP literal.
    assert validate_download_url("https://1.1.1.1/model.gguf", allow_private=False)


def test_parse_parameters():
    parsed = parse_parameters(
        "temperature 0.7\n"
        "num_ctx 4096\n"
        "# a comment\n"
        'stop "<|im_end|>"\n'
        "stop <|eot|>\n"
        "top_p: 0.9\n"
    )
    assert parsed["temperature"] == 0.7
    assert parsed["num_ctx"] == 4096
    assert parsed["top_p"] == 0.9
    assert parsed["stop"] == ["<|im_end|>", "<|eot|>"]


def test_parse_parameters_rejects_unknown_key():
    with pytest.raises(ValidationError):
        parse_parameters("evil_directive 1")


@pytest.mark.parametrize("key", ["mirostat", "mirostat_tau", "tfs_z", "penalize_newline"])
def test_parse_parameters_rejects_removed_ollama_keys(key):
    with pytest.raises(ValidationError):
        parse_parameters(f"{key} 1")


def test_parse_parameters_rejects_garbage_line():
    with pytest.raises(ValidationError):
        parse_parameters("temperature")
