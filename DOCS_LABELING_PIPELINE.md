# Data Labeling Pipeline — Documentation

This document describes the **YouTube comment data labeling pipeline** added since the last git push (commit: *Include like counts in THREAD output format*). It covers design, usage, and implementation details.

---

## 1. Overview

The pipeline takes **formatted comment threads** (from `youtube_comments_extractor.py` with `--format threads`) and sends them to **DeepSeek-V3.2 (non-thinking mode)** via the [DeepSeek API](https://api.deepseek.com) to extract **structured audience signals** per the schema in `systemPrompt.xml`. Outputs are saved as JSON with each label explicitly tied to its `thread_id`.

### Main components

| Component | Purpose |
|-----------|--------|
| **label_comments.py** | Main script: parse threads → batch → call API → map signals to threads → save JSON |
| **systemPrompt.xml** | System prompt defining input format, signal axes, and strict JSON-only output |
| **experiment_consistency.py** | Runs the labeler multiple times and measures token throughput and output consistency |
| **schema** | Reference for thread format (`THREAD <id>`, `local_id\|parent_local_id\|like_count\|text`) |

---

## 2. Thread format and batching

- **Input format** (see `schema`):  
  - One thread per root comment: `THREAD <thread_id>` followed by lines `local_id|parent_local_id|like_count|text`.  
  - Threads separated by blank lines.  
  - `local_id` starts at 1 per thread.

- **Batching rules** (no thread splitting, no mixing):
  - **Input cap:** 7,000 tokens per request (configurable). System prompt is subtracted; remaining budget is filled with whole threads only.
  - **Output cap:** Batches are also limited by **estimated output size** so the API response is not truncated. The DeepSeek API returns at most 4K tokens by default, 8K max (see [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)). Each thread’s signals use ~140+ tokens. So we cap **threads per batch** to `max_output_tokens / avg_output_tokens_per_thread` (default 8000/150 ≈ 53). This prevents large batches of one-line threads from producing truncated JSON and empty signals for most threads.
  - If a single thread exceeds the input budget, it is sent in its own batch (with a warning).

- **Output mapping:**  
  API returns one JSON object per batch with a `signals` array. Signals are mapped back to threads using `source_comments` (local_ids). Each thread’s signals are written with that thread’s `thread_id`; every signal object in the saved JSON also includes a **`thread_id`** field so labels are always tied to their comment thread.

---

## 3. API and robustness

- **Endpoint:** `POST https://api.deepseek.com/chat/completions`  
- **Model:** `deepseek-chat` (non-thinking).  
- **Auth:** `Authorization: Bearer <DEEPSEEK_API_KEY>` (from `.env`).  
- **Request:** `system` message = contents of `systemPrompt.xml`; `user` message = video description (optional) + batch of threads.  
- **Response:** `response_format: { "type": "json_object" }`; we parse the `content` of the first choice.

**Robustness:**

- **JSON parsing:**  
  If the model returns malformed or truncated JSON, we try: strip markdown code fences, remove trailing commas, truncate at last complete `}` and close brackets, then close any remaining open brackets. See `parse_json_robust()` in `label_comments.py`.

- **Parse-failure retries:**  
  If parsing fails, we **re-call the API** for that batch (up to **2 retries**, i.e. 3 attempts total) before recording an error for that batch.

- **Network/API retries:**  
  Failed requests (timeouts, 5xx, etc.) are retried with exponential backoff (2s, 4s, 8s), up to 3 times.  
  [Rate limit](https://api-docs.deepseek.com/quick_start/rate_limit): no strict rate limit; under load the server may send empty lines or keep-alive; connection may close after 10 minutes.

---

## 4. Idempotency and progress

- **Idempotency:**  
  Before processing, we load the existing output file (if any). For each batch, if **all** threads in that batch already appear in the loaded results, the batch is **skipped** (no API call). So re-runs only process new or previously failed threads.

- **Progress and resume:**  
  Optional `--progress-file` and `--resume`: progress is saved so that after an interrupt you can resume and only process remaining batches.

- **Incremental saves:**  
  After each batch, results are merged and written to the output file (with a lock). So partial results are preserved if the process stops.

- **Retry for empty signals:**  
  If many threads in a batch get **empty signals** (e.g. because the response was truncated), the script can **retry** those threads in smaller batches. Use `--retry-empty-signals`. Retries run up to 2 times; only threads with empty signals are re-sent, with the same output-token cap so responses fit. After the first pass, the script prints how many threads have empty signals and suggests `--retry-empty-signals` if needed.

---

## 5. Token usage and throughput

- **Usage:**  
  Each API response’s `usage` (prompt_tokens, completion_tokens, total_tokens) is summed across batches.

- **After a run:**  
  The script prints:  
  `Token usage: prompt=X completion=Y total=Z | elapsed=Ts | throughput=N tokens/s`

- **Optional stats file:**  
  With `--stats-file <path>`, a JSON file is written with:
  - `prompt_tokens`, `completion_tokens`, `total_tokens`
  - `elapsed_sec`, `tokens_per_sec`
  - `batches_processed`, `threads_processed`

---

## 6. LLM trace logging (debugging)

- **Option:** `--llm-log-dir <DIR>`  
- **Behavior:**  
  For each batch (and each retry attempt), a **trace** file is written:  
  `DIR/batch_<index>_attempt_<n>_<YYYYMMDD>_<HHMMSS>.txt`  
  (timestamp is UTC when the log is written).  
  Each file contains:
  - **Meta (JSON):** batch_index, attempt, thread_ids, timestamp_utc, parse_ok, content_length, completion_tokens, prompt_tokens, total_tokens (when available), and parse_error (if any).
  - **---SENT_TO_LLM---:** full request: `---system_prompt---` (system prompt text) and `---user_content---` (batch threads + optional video description).
  - **---RECEIVED_FROM_LLM---:** raw model response (exactly as returned).

This gives a full trace of what was sent and received for each API call, for debugging and auditing.

---

## 7. System prompt (strict output)

- **File:** `systemPrompt.xml`  
- **Role:** Analyst extracting structured audience signals from YouTube comment threads.  
- **Input:** Short video description + multiple threads in the schema format.  
- **Output rules (strict):**
  - Reply must be **exactly one valid JSON object** (starts with `{`, ends with `}`).
  - **No** markdown (no ```json or ```).
  - **No** text or commentary before or after the JSON.
  - Do not truncate strings; escape special characters.
  - Use only the schema keys; `source_comments` = array of integers (local_id); `topics` = array of strings; `affirmation_flag` = boolean.

- **Signal axes** (see `systemPrompt.xml`):  
  signal_type, topics, emotion, emotion_intensity, intent_clarity, specificity, local_prevalence_hint, engagement_depth, consensus_state, affirmation_flag, signal_strength_hint, source_comments.

---

## 8. Output JSON shape

- **Top level:** `{ "threads": [ ... ] }`  
- **Each thread:** `{ "thread_id": "<id>", "signals": [ ... ] }`  
- **Each signal:** All schema fields **plus** `"thread_id": "<id>"` so every label is explicitly tied to its comment thread.  
- **source_comments:** List of `local_id` values (within that thread) that support the signal.

Example:

```json
{
  "threads": [
    {
      "thread_id": "Ugzc_3sN1bhzM8hlue54AaABAg",
      "signals": [
        {
          "signal_type": "affirmation",
          "topics": ["heroic fantasy", "moral clarity"],
          "emotion": "enjoyment",
          "thread_id": "Ugzc_3sN1bhzM8hlue54AaABAg",
          "source_comments": [1, 8, 10]
        }
      ]
    }
  ]
}
```

---

## 9. Experiment script (consistency and throughput)

- **Script:** `experiment_consistency.py`  
- **Purpose:** Run the labeler **multiple times** on the same input and report:
  - **Throughput:** Min/max/mean tokens per second (from each run’s `--stats-file`).
  - **Consistency:** Pairwise similarity between runs:
    - **Mean Jaccard** over threads (signal sets compared by signal_type + topics + emotion).
    - **Exact-match ratio:** Fraction of threads where the signal set is identical between two runs.

**Usage:**

```bash
python experiment_consistency.py --threads-file test.txt --runs 3 --out-dir experiment_runs
```

**Outputs in `--out-dir`:**

- `run_001.json`, `run_002.json`, … — labeled outputs.
- `run_001_stats.json`, … — token and timing stats per run.
- `report.json` — consistency metrics and throughput summary.

---

## 10. Command-line reference (label_comments.py)

| Argument | Description |
|----------|-------------|
| `--threads-file` | Input threads file (required). |
| `--output` / `-o` | Output JSON path (required). |
| `--system-prompt` | Path to system prompt XML (default: systemPrompt.xml). |
| `--api-key` | DeepSeek API key (default: from DEEPSEEK_API_KEY). |
| `--max-tokens` | Max **input** context tokens per batch (default: 7000). |
| `--max-output-tokens` | Max **response** tokens per request; batches are sized so output fits (default: 8000; API max 8K). |
| `--avg-output-tokens-per-thread` | Estimate used to cap threads per batch (default: 150). |
| `--retry-empty-signals` | Retry threads that got empty signals in smaller batches (up to 2 retries). |
| `--max-parallel` | Max concurrent API requests (default: 20). |
| `--video-description` | Optional video description for the prompt. |
| `--progress-file` | Progress checkpoint path (for resume). |
| `--resume` | Resume from progress file if present. |
| `--llm-log-dir` | Directory to log raw LLM responses (debugging). |
| `--stats-file` | Write token/timing stats to this JSON file. |

**Example:**

```bash
python label_comments.py \
  --threads-file threads_JkwMMpz699Y.txt \
  --output labeled.json \
  --video-description "Video about fantasy shows..." \
  --llm-log-dir llm_outputs \
  --stats-file run_stats.json \
  --max-parallel 10 \
  --retry-empty-signals
```

---

## 11. Files touched / added (since last push)

| File | Status / role |
|------|----------------|
| **label_comments.py** | New: main labeling pipeline (parse, batch, API, map, save, token stats, logging). |
| **systemPrompt.xml** | New: system prompt for DeepSeek (strict JSON output, signal schema). |
| **experiment_consistency.py** | New: multi-run experiment for throughput and consistency. |
| **.gitignore** | Updated: ignore generated outputs and logs (e.g. llm_outputs/, test_labeled.json, experiment_runs/). |
| **DOCS_LABELING_PIPELINE.md** | New: this documentation. |

Generated or local data (typically not committed): `llm_outputs/`, `test_labeled.json`, `experiment_runs/`, `*.progress.json`, `test.txt` (optional test input).

---

## 12. Dependencies

- **requests** — HTTP calls to DeepSeek API.  
- **python-dotenv** — Load `DEEPSEEK_API_KEY` from `.env`.  

No new dependencies beyond the existing project.
