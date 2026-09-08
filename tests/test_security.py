import pytest

from app.security import (
    MAX_OPTION_TEXT_LEN,
    MAX_PARAM_LINES,
    MAX_STOP_SEQUENCES,
    QUANT_LEVELS,
    ValidationError,
    addr_is_blocked,
    model_name_from_hf_repo,
    model_name_from_url,
    parse_hf_repo_id,
    parse_hf_revision,
    parse_parameters,
    require_namespaced_model_name,
    validate_download_url,
    validate_model_name,
    validate_option_text,
    validate_quant_level,
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


def test_parse_parameters_rejects_oversized_block():
    with pytest.raises(ValidationError):
        parse_parameters("x" * (MAX_OPTION_TEXT_LEN + 1))


def test_parse_parameters_rejects_too_many_lines():
    with pytest.raises(ValidationError):
        parse_parameters("\n".join(["temperature 0.7"] * (MAX_PARAM_LINES + 1)))


def test_parse_parameters_caps_stop_sequences():
    with pytest.raises(ValidationError):
        parse_parameters("\n".join([f"stop x{i}" for i in range(MAX_STOP_SEQUENCES + 1)]))


def test_validate_option_text():
    assert validate_option_text("  hi  ", "System prompt") == "hi"
    with pytest.raises(ValidationError):
        validate_option_text("x" * (MAX_OPTION_TEXT_LEN + 1), "System prompt")


@pytest.mark.parametrize(
    "addr",
    ["127.0.0.1", "10.1.2.3", "192.168.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1"],
)
def test_addr_is_blocked_private(addr):
    assert addr_is_blocked(addr)


@pytest.mark.parametrize("addr", ["1.1.1.1", "8.8.8.8", "not-an-ip"])
def test_addr_is_blocked_allows_public(addr):
    assert not addr_is_blocked(addr)


# --------------------------------------------------------------------------- #
# HF -> GGUF conversion page
# --------------------------------------------------------------------------- #
def test_parse_hf_repo_id_bare():
    assert parse_hf_repo_id("huihui-ai/Huihui-Ornith-1.5-9B-abliterated") == \
        "huihui-ai/Huihui-Ornith-1.5-9B-abliterated"


def test_parse_hf_repo_id_from_url():
    assert parse_hf_repo_id(
        "https://huggingface.co/huihui-ai/Huihui-Ornith-1.5-9B-abliterated/tree/main"
    ) == "huihui-ai/Huihui-Ornith-1.5-9B-abliterated"


@pytest.mark.parametrize("text", [
    "not-a-repo", "owner/../etc", "", "owner//repo", "a/b/c",
    "https://evil.example.com/owner/repo",
])
def test_parse_hf_repo_id_rejects_bad_input(text):
    with pytest.raises(ValidationError):
        parse_hf_repo_id(text)


def test_parse_hf_revision_defaults_to_main():
    assert parse_hf_revision("") == "main"
    assert parse_hf_revision("v1.0") == "v1.0"


@pytest.mark.parametrize("rev", ["../etc/passwd", "a b", ""])
def test_parse_hf_revision_rejects_bad_input(rev):
    if rev == "":
        assert parse_hf_revision(rev) == "main"
    else:
        with pytest.raises(ValidationError):
            parse_hf_revision(rev)


def test_model_name_from_hf_repo():
    assert model_name_from_hf_repo("huihui-ai/Huihui-Ornith-1.5-9B-abliterated") == \
        "huihui-ornith-1.5-9b-abliterated"


def test_validate_quant_level():
    assert validate_quant_level("q4_k_m") == "Q4_K_M"
    assert "F16" in QUANT_LEVELS
    with pytest.raises(ValidationError):
        validate_quant_level("Q9_BOGUS")


def test_require_namespaced_model_name():
    require_namespaced_model_name("someone/mymodel")
    require_namespaced_model_name("someone/mymodel:latest")
    with pytest.raises(ValidationError):
        require_namespaced_model_name("mymodel")
    with pytest.raises(ValidationError):
        require_namespaced_model_name("mymodel:latest")
