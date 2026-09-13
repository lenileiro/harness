# Media tools

Image/audio/file attachments are versioned core data, retained in sessions and tool results. The CLI accepts repeated `run --attach PATH`; chat `/attach PATH` attaches a file to the next turn. `MediaAttachment` holds bounded inline base64 or an explicit HTTP(S) URL. Model-visible media is checked against the selected adapter before execution; unsupported providers fail explicitly.

Audio can carry a positive `duration_ms` value (at most 24 hours). Local audio
loading and generated speech derive it from actual bounded bytes with
[TinyTag](https://github.com/tinytag/tinytag), without fetching URLs or running an
external decoder. Unknown formats leave duration unset. Hosted-audio callers must
supply an accurate duration when their channel requires it; no default playback
length is invented. Duration survives session and channel-outbox persistence.

To enable local media workflows:

```toml
[media]
enabled = true
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
image_model = "YOUR_IMAGE_MODEL"
transcription_model = "YOUR_TRANSCRIPTION_MODEL"
speech_model = "YOUR_SPEECH_MODEL"
voice = "YOUR_VOICE"
```

Select model IDs supported by your provider. Omit any model setting to omit that service tool. `read_media` reads a bounded workspace image, audio file or document without an API request. `image_generate`, `audio_transcribe` and `speech_generate` use normal Harness approvals before their external requests. This configuration is local-agent-only and does not expand remote API/gateway access.

Generation uses OpenAI-compatible image/audio endpoints. The image endpoint must return `b64_json`; `image_parameters` can set compatible provider fields such as `response_format = "b64_json"` when required. `output_format` is restricted to png/jpeg/webp and determines the stored MIME type and extension. Returned image URLs are not followed. Speech produces MP3. Generated speech is marked as a deliverable rather than model-visible input so a text-only model can still return it to the user.

Input/output defaults to a 20 MiB limit. Workspace traversal and private `.harness` paths are rejected. HTTP requests use explicit credentials, bounded streams, no redirects/proxy inheritance and no automatic replay after ambiguous failure. Generated artifacts have unique names under `artifacts/media`; their attachments remain usable after the tool context closes.

The OpenAI-compatible, Anthropic, Ollama and Codex app-server adapters convert only the media types they advertise. Named compatible providers configure their own `input_media` capability list. Messaging delivery support is tracked separately in the [channel inventory](../../docs/channel-parity.md).

Offline tests exercise actual wire request shapes, capability boundaries, storage/resume and artifact files. No live image/audio provider was invoked. [Portal account routes](../../docs/accounts-and-services.md) can supply the same services with their managed account credentials.
