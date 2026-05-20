# harness-adapter-ollama

[Ollama](https://ollama.ai/) adapter for Harness. Streams chat completions via Ollama's `/api/chat` endpoint (also supports the OpenAI-compatible `/v1/chat/completions`).

Defaults to `http://localhost:11434`; override with `OLLAMA_HOST` or per-session config.
