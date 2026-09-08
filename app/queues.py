"""Job-queue name constants shared between the web app and both worker
images. Kept dependency-free so app.main (light) can import it without
pulling in anything from the heavy converter image, and vice versa."""

GGUF_QUEUE = "gguf_download"
GGUF_HEALTH_KEY = "gguf_download:health-check"
HF_CONVERT_QUEUE = "hf_convert"
HF_CONVERT_HEALTH_KEY = "hf_convert:health-check"
