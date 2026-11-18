"""Tiny sentence-embedding API, meant to run as a free Hugging Face Space (Gradio SDK, 16 GB RAM).

The WhatsApp bot (NLU_BACKEND=remote) sends text here and gets back one meaning vector per text.
The model (BAAI/bge-small-en-v1.5, ONNX, no PyTorch) never has to fit in the bot's own small server.

    POST /embed   {"texts": ["my inverter beeps", ...]}   header  Authorization: Bearer <EMBED_API_KEY>
               -> {"model": "...", "dim": 384, "vectors": [[...], ...]}
    GET  /health  -> {"ok": true, "model": "..."}
"""
import hmac
import os

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastembed import TextEmbedding

MODEL = os.getenv("NLU_MODEL", "BAAI/bge-small-en-v1.5")
KEY = os.getenv("EMBED_API_KEY", "")
MAX_TEXTS = 256
MAX_CHARS = 1000

model = TextEmbedding(model_name=MODEL, cache_dir=os.getenv("MODEL_CACHE", "models"))
api = FastAPI()


@api.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@api.post("/embed")
async def embed(request: Request):
    if not KEY:
        return JSONResponse({"error": "EMBED_API_KEY is not set on the server"}, status_code=503)
    sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(sent, KEY):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        texts = (await request.json()).get("texts")
    except Exception:
        texts = None
    if not isinstance(texts, list) or not texts or len(texts) > MAX_TEXTS or not all(isinstance(t, str) for t in texts):
        return JSONResponse({"error": f"send {{'texts': [1..{MAX_TEXTS} strings]}}"}, status_code=400)
    vectors = [v.tolist() for v in model.embed([t[:MAX_CHARS] for t in texts])]
    return {"model": MODEL, "dim": len(vectors[0]), "vectors": vectors}


with gr.Blocks(title="Solar bot embeddings") as page:
    gr.Markdown("# Solar bot embeddings\nThis Space only serves the bot's `/embed` API.")

app = gr.mount_gradio_app(api, page, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
