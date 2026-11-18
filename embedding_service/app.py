"""Tiny sentence-embedding API, meant to run as a free Hugging Face Space (Gradio SDK, ZeroGPU or CPU).

The WhatsApp bot (NLU_BACKEND=remote) sends text here and gets back one meaning vector per text.
The model (BAAI/bge-small-en-v1.5, ONNX, no PyTorch) never has to fit in the bot's own small server.

Gradio serves everything, so the Space starts the way Hugging Face expects. The endpoint is Gradio's API:

    POST /gradio_api/call/embed   {"data": ["<json list of texts>", "<EMBED_API_KEY>"]}  -> {"event_id": "..."}
    GET  /gradio_api/call/embed/<event_id>   (server-sent events; the "complete" event carries
                                              ["<json>"] = {"model": ..., "dim": 384, "vectors": [[...], ...]})
"""
import hmac
import json
import os

import gradio as gr
from fastembed import TextEmbedding

try:
    # ZeroGPU Spaces refuse to start unless the app declares one @spaces.GPU function. This one is never
    # called: the embeddings below run on the CPU. On ordinary CPU hardware the import simply fails, which is fine.
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return 0
except ImportError as exc:
    print(f"NOTE: the 'spaces' package is not available ({exc}); running without a ZeroGPU placeholder", flush=True)

MODEL = os.getenv("NLU_MODEL", "BAAI/bge-small-en-v1.5")
KEY = os.getenv("EMBED_API_KEY", "")
MAX_TEXTS = 256
MAX_CHARS = 1000

model = TextEmbedding(model_name=MODEL, cache_dir=os.getenv("MODEL_CACHE", "models"))


def embed(texts_json: str, key: str) -> str:
    if not KEY:
        raise gr.Error("EMBED_API_KEY is not set on the server")
    if not hmac.compare_digest((key or "").strip().encode(), KEY.encode()):
        raise gr.Error("unauthorized")
    try:
        texts = json.loads(texts_json)
    except Exception:
        texts = None
    if not isinstance(texts, list) or not texts or len(texts) > MAX_TEXTS or not all(isinstance(t, str) for t in texts):
        raise gr.Error(f"send a JSON list of 1..{MAX_TEXTS} strings")
    vectors = [v.tolist() for v in model.embed([t[:MAX_CHARS] for t in texts])]
    return json.dumps({"model": MODEL, "dim": len(vectors[0]), "vectors": vectors})


with gr.Blocks(title="Solar bot embeddings") as demo:
    gr.Markdown("# Solar bot embeddings\nThis Space only serves the bot's `embed` API.")
    gr.api(embed, api_name="embed")

if __name__ == "__main__":
    demo.launch()
