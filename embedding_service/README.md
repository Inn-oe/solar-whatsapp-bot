# Embedding service for the solar WhatsApp bot

A small API that turns text into meaning vectors (BAAI/bge-small-en-v1.5). The bot calls it when
`NLU_BACKEND=remote`, so the model runs here (free, 16 GB RAM) instead of inside the bot's 512 MB server.

Deploy as a Hugging Face Space with the **Gradio** SDK (the Docker SDK is paid): upload `app.py` and
`requirements.txt` only (keep the README the Space generated). Add the secret `EMBED_API_KEY` (any long random
string) and use the same value as `NLU_REMOTE_KEY` on the bot.
