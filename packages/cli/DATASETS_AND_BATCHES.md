# Offline datasets and durable batches

`harness dataset` prepares actual trajectory datasets for external training tools. It never starts a training job or downloads data/models/tokenizers. The output is inspectable JSONL; importing it does not create executable sessions or transfer approval grants into the runtime.

```sh
harness dataset export SESSION_ID --workspace /project --output session.jsonl
harness dataset import public-chat.jsonl --format trl --output archive.jsonl
harness dataset validate archive.jsonl --complete
harness dataset compress archive.jsonl --max-tokens 16000 --output compressed.jsonl
harness dataset prepare compressed.jsonl --format trl --seed 42 --eval-fraction 0.1 --output training-data
```

Local session export uses the selected profile's database by default. `--database` selects another local database. The saved workspace and optional `--user` identity must match; unscoped legacy sessions are not exported through this command. Known secret patterns are redacted, but attachments and arbitrary user content remain part of the archive and need review before sharing. Batch/API export uses the service's persisted owner identity.

## Archives and validation

Each JSONL row has `format: "harness.trajectory"`, `version: 1`, `messages`, and optional `session`, `tools`, `runs`, `events`, and `metadata`. This is the same format exported by the HTTP server. Media attachments and original tool-call/result IDs survive an archive round trip. Unsupported versions/fields, malformed arguments, orphan results, duplicate IDs, mismatched tool names and interleaved unfinished tool groups fail validation.

Unresolved calls at the end of an interrupted/paused trajectory can be archived. `validate --complete`, compression, and training preparation require complete pairs. Training records also require a user request and a completed assistant answer. Text training formats reject media instead of silently removing it.

Input adapters accept OpenAI chat JSONL, Hugging Face TRL conversational JSONL, and ShareGPT `conversations` with `from`/`value`. OpenAI arguments are JSON strings; TRL/ShareGPT arguments are JSON objects. Missing tool-call IDs can be assigned deterministically only when results can be matched unambiguously. Unsupported columns or message fields require explicit conversion.

Tool-call training needs the original tool schemas. Supply a public function-schema array with `--tools tools.json` when importing, exporting or preparing data. Schemas are never guessed from observed arguments. The importer validates the function envelope, unique names and object parameters; downstream trainers remain responsible for model-specific schema constraints.

## Compression and preparation

Compression is deterministic and records every omitted message index and shortened tool-result ID in `metadata.compression`. It shortens long tool output first, then removes whole older message blocks. An assistant call and all of its parallel results always form one block. It preserves system instructions, the first/last user request and the final message. If that protected content, schemas or attachments cannot fit, it fails without publishing partial output.

`--max-tokens` uses a documented **UTF-8 JSON bytes / 4 estimate**, including schemas and media. It is not an exact model-tokenizer count or a promise that an external trainer's context window will fit. No tokenizer network fetch or remote code execution is involved.

Preparation deduplicates conversations independently of session metadata and incidental tool IDs, sorts them deterministically, applies seeded sampling, then partitions disjoint training/evaluation records. `--sample` refuses to oversample. `training-data/` contains `train.jsonl`, `eval.jsonl` and `manifest.json` with source fingerprints, split membership, seed and compression reports. A single unique record yields no evaluation examples. Output files/directories have private permissions and existing outputs are never overwritten. Input is bounded to 128 MiB, 24 MiB per record and 10,000 records.

The public conversational/tool schema follows [Hugging Face TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats). The preparation workflow corresponds to the data sampling/compression role of [Hermes's sample_and_compress.py](https://github.com/NousResearch/hermes-agent/blob/main/scripts/sample_and_compress.py), while requiring an explicitly supplied local dataset.

## Durable batch execution

```sh
# requests.jsonl: one RunSubmission object per line, e.g. {"prompt":"Inspect the project"}
harness batch --workspace /project submit requests.jsonl --queue-only
harness batch --workspace /project status BATCH_ID
harness batch --workspace /project work
harness batch --workspace /project export BATCH_ID --output trajectories.jsonl
harness batch --workspace /project cancel BATCH_ID
harness batch --workspace /project resume RUN_ID --prompt "Continue from saved state"
```

Batches use the same SQLite queue, identity, approvals, cancellation and restart recovery as the HTTP service. The default database is `/project/.harness/server.db`; `--database` lets a CLI operate on the same explicitly chosen server database. `--owner` names the trusted local database identity. Read/status/cancel/queue-only operations do not construct adapters or call models. All input rows validate before anything is queued.

Foreground submit waits for persisted terminal results. `work` drains the database queue; one executing process may own that database at a time. An active server can execute queue-only submissions while a CLI performs status/cancellation operations. Restart marks interrupted running work for inspection rather than silently replaying it. Resume takes an explicit run ID and rejects completed runs, which require a new submission. Approval-paused work requires resolving the approval through the API/UI or existing approval CLI before continuation. No batch path bypasses Harness tool gates.

Batch exports include whole owned sessions, with calls and results kept together. Active batches must finish or be cancelled before export; a cancelled run that never created a session has no trajectory to export. Exported tool datasets still need their original schemas before training preparation.
