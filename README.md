# youtubeComments — Fast YouTube Comment Thread Extractor

This repo contains a **YouTube comments extractor** optimised for:

- **Speed**: parallel fetching of replies + round‑robin use of multiple API keys.
- **LLM‑friendly format**: a compact, token‑efficient thread schema for downstream analysis.

It uses the **YouTube Data API v3** to fetch:

- All **top‑level comments** for a video.
- All **direct replies** to each top‑level comment.
- A **virtual deep thread tree** inferred from `@mentions` so you can treat the discussion as having more than two levels of depth.

---

## 1. Features

- **Full thread coverage**
  - Fetches every root comment and all its replies.
  - Handles pagination for both `commentThreads.list` and `comments.list`.

- **Multi‑key parallelism**
  - Supports multiple API keys (`YOUTUBE_API_KEY_1`, `YOUTUBE_API_KEY_2`, …).
  - Uses a thread‑safe round‑robin key pool to:
    - Spread quota load across keys.
    - Issue many reply‑fetch requests **in parallel**.

- **Virtual deep threading**
  - YouTube’s API only exposes: **root comment** + **direct replies**.
  - This tool builds a **virtual hierarchy with >2 levels** by:
    - Inspecting `@mentions` in reply text (e.g. `@raining_macondo`).
    - Normalising author names and mention tokens.
    - Attaching each reply to the **most recent earlier comment** whose author matches the mention.
  - If no good match is found, the reply is attached directly to the root.

- **Compact THREAD schema for LLMs**
  - Output format designed to minimise tokens while preserving structure.
  - Depth‑first pre‑order traversal with chronological sibling ordering.

---

## 2. Installation

### 2.1. Clone the repo

```bash
git clone https://github.com/bigtopro/youtubeComments.git
cd youtubeComments
```

### 2.2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

### 2.3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configure API keys

Create a `.env` file **in the repo root** (do not commit it) with one or more YouTube Data API keys:

```bash
touch .env
```

Example `.env`:

```env
YOUTUBE_API_KEY_1=YOUR_FIRST_API_KEY
YOUTUBE_API_KEY_2=YOUR_SECOND_API_KEY
YOUTUBE_API_KEY_3=YOUR_THIRD_API_KEY
```

Rules:

- At least **one** API key is required.
- Any environment variable whose name starts with `YOUTUBE_API_KEY` and has a non‑empty value will be used.
- Keys are placed in a **round‑robin pool** and used across all API calls.

You can also pass a primary key via `--api-key`; it will be added to the pool as well.

---

## 4. Usage

The main entrypoint is `youtube_comments_extractor.py`.

### 4.1. Basic usage (JSON output)

```bash
source .venv/bin/activate

python youtube_comments_extractor.py \
  --video-id rPckDuTaSnw \
  --format json \
  --output comments.json
```

This writes a structured JSON file:

- One object per **root comment**, with:
  - `id`, `videoId`, `author`, `text`, `likeCount`, `publishedAt`, `updatedAt`
  - `replies`: list of direct replies (as returned by the API).

### 4.2. LLM‑ready THREAD format

Use the compact, token‑efficient format designed for LLM analysis:

```bash
python youtube_comments_extractor.py \
  --video-id rPckDuTaSnw \
  --format threads \
  --output threads.txt
```

Each thread (one root comment) is represented as:

```text
THREAD <thread_id>
<local_id>|<parent_local_id>|<like_count>|<text>
...
```

Example:

```text
THREAD UgzHy6Mw2e3oc_GVYyl4AaABAg
1|0|12|Root
2|1|5|Reply A
3|2|0|Reply B
4|2|3|Reply C
5|4|7|Reply D
```

#### Field meanings

- **THREAD line**
  - `THREAD <thread_id>`
  - `thread_id` is the original **YouTube top‑level comment ID**.

- **Comment lines**
  - `<local_id>`: integer, starts at 1 per thread, unique within thread.
  - `<parent_local_id>`:
    - `0` for the root comment.
    - Otherwise: `local_id` of the immediate parent.
  - `<like_count>`: integer like count from the YouTube API (0 if missing).
  - `<text>`: single‑line comment text (Unicode + emoji preserved; internal newlines collapsed to spaces).

#### Ordering guarantees

Within each thread:

1. Root comment is always first:

   ```text
   1|0|<root_text>
   ```

2. Traversal is **depth‑first pre‑order**:
   - A parent always appears before its children.

3. Siblings are sorted by:

   ```text
   publishedAt ASC
   ```

Threads are separated by a single blank line.

---

## 5. CLI options

```bash
python youtube_comments_extractor.py --help
```

Key arguments:

- `--video-id` (**required**): YouTube video ID (`v` parameter in the URL).
- `--api-key`: primary API key (optional if provided in env).
- `--output` / `-o`: output file path. If omitted, prints to stdout.
- `--format`:
  - `json` (default): full JSON.
  - `threads`: compact THREAD schema for LLMs.
- `--indent`: JSON indentation (only used with `--format json`).
- `--page-size`: batch size per API call (1–100, default 100).

---

## 6. How the deep threading heuristic works

Because the YouTube API only models:

- Root comments, and
- Direct replies to root,

this tool builds a **virtual multi‑level tree** using a heuristic:

1. For each root, collect all replies and sort them by `publishedAt` ascending.
2. For each reply:
   - Extract `@mentions` from the text (e.g. `@raining_macondo`).
   - Normalise both mention tokens and author names:
     - Lowercase.
     - Remove non‑alphanumeric characters.
   - Search backwards through earlier comments in the same thread
     (root + previous replies) and find the **most recent** one
     whose normalised author string contains the mention token.
   - Attach the reply as a child of that comment.
3. If no suitable parent is found, attach the reply directly to the root.

This produces a **best‑effort deeper hierarchy** where:

- Explicit `@mentions` control the inferred parent/child relationships.
- Chronology breaks ties (preference to the most recent eligible parent).
- The resulting tree can have **arbitrary depth** for analysis purposes.

---

## 7. Example end‑to‑end flow

```bash
git clone https://github.com/bigtopro/youtubeComments.git
cd youtubeComments

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo 'YOUTUBE_API_KEY_1=YOUR_API_KEY_HERE' > .env

python youtube_comments_extractor.py \
  --video-id rPckDuTaSnw \
  --format threads \
  --output threads_rPckDuTaSnw.txt
```

You can then feed `threads_rPckDuTaSnw.txt` directly to an LLM (or chunk by `THREAD` blocks) to analyse the full discussion while minimising token usage.

