from app.util import flatten_info, human_bytes


def test_human_bytes():
    assert human_bytes(None) == "—"
    assert human_bytes(512) == "512 B"
    assert human_bytes(1536) == "1.50 KB"
    assert human_bytes(5 * 1024**3) == "5.00 GB"


def test_flatten_info():
    rows = dict(
        flatten_info(
            {
                "details": {"family": "llama", "quantization_level": "Q4_K_M", "unused": "x"},
                "model_info": {
                    "llama.context_length": 8192,
                    "general.parameter_count": 8030000000,
                    "general.architecture": "llama",
                },
                "digest": "sha256:abc",
            }
        )
    )
    assert rows["family"] == "llama"
    assert rows["quantization_level"] == "Q4_K_M"
    assert rows["llama.context_length"] == "8192"
    assert rows["general.parameter_count"] == "8030000000"
    assert "general.architecture" not in rows
    assert rows["digest"] == "sha256:abc"
