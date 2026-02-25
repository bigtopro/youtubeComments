Perfect. I’ll give you a **production-grade coding agent prompt** that you can drop into Cursor / Claude Code / GPT-o / whatever coding assistant you’re using.

This prompt assumes:

* You already have canonical topics per video
* You want to build the macro topic labeller pipeline
* You’re using an LLM API (OpenRouter / OpenAI compatible)
* You want batching + validation + retry logic

This is not just “write some code”.
This defines architecture, modules, and constraints.

---

# 🧠 CODING AGENT PROMPT

## Macro Topic Labeller Pipeline

---

You are building a production-ready Macro Topic Labeller module for a YouTube comment intelligence system.

The goal is to transform thousands of canonical topic strings into stable macro topic labels using an LLM in batched calls.

The implementation must be:

* Deterministic
* Token-efficient
* Validated
* Retry-safe
* Extensible

---

# 🏗️ SYSTEM REQUIREMENTS

We already have:

```
canonical_topics: List[str]
```

Example:

```
[
  "CIA funding",
  "CIA propaganda",
  "Bangladesh election",
  ...
]
```

We need to build:

```
canonical_topic → global_macro_topic
```

---

# 🔷 PHASE 1 — Batch Macro Grouping

## Step 1: Chunk Canonical Topics

* Chunk size: 200 topics per batch
* Each topic must be numbered starting from 1 per batch
* Maintain mapping: batch_index → original_topic

Example batch input format:

```
Video Title: {title}
Video Description: {description}

#TOPICS
1. CIA funding
2. CIA propaganda
3. Bangladesh election
...
#END
```

---

## Step 2: LLM System Prompt

Use a constant system prompt string (do NOT dynamically generate):

```
You are a semantic taxonomy builder.

You will receive a numbered list of canonical topic labels extracted from YouTube comments for a single video.

Your task is to group them into broader macro topics.

Definition of a macro topic:
- A stable, reusable semantic anchor.
- May be a person, product, feature, concept, event, organization, country, technology, or clearly defined domain.
- Must be concrete and meaningful.
- Must not be vague or overly abstract.
- Must not merge clearly unrelated subjects.

Rules:
1. Each topic number must appear exactly once.
2. Every topic must belong to exactly one macro topic.
3. Do not invent subjects not grounded in the input.
4. Do not create vague labels like:
   - "General discussion"
   - "Various topics"
   - "Other issues"
5. Keep macro labels concise (1–4 words).
6. If a topic is already broad and cannot logically group with others, it may form its own macro topic.

Output format (strict):

macro_label: n1,n2,n3
macro_label: n4,n5
...

Only output macro_label: numbers.
No explanations.
```

---

## Step 3: LLM Call

* temperature = 0.2
* max_output_tokens ≈ 3500
* retry up to 3 times if validation fails

---

## Step 4: Parse Output

Expected output example:

```
CIA: 1,2,5
Bangladesh: 3,4
```

Implement parser that:

* Splits lines
* Extracts macro_label
* Extracts list of integers
* Validates that:

  * All numbers 1..N appear exactly once
  * No duplicates
  * No missing numbers
  * No out-of-range numbers

If validation fails:

* Retry LLM call

If still fails after 3 attempts:

* Log error
* Fallback to identity mapping (each topic its own macro)

---

## Step 5: Convert to Canonical Mapping

After parsing:

```
batch_macro_mapping: Dict[str, List[str]]
```

Map numbers back to original topic strings.

Store:

```
canonical_topic → batch_macro_label
```

Collect all batch_macro_labels across batches.

---

# 🔷 PHASE 2 — Global Macro Deduplication

After Phase 1 completes:

You will have:

```
all_batch_macro_labels: List[str]
```

Some may be duplicates or near-duplicates.

---

## Step 1: Unique Macro Labels

Get unique macro labels across batches.

Number them:

```
#MACROS
1. British colonialism
2. British colonial legacy
3. CIA
4. CIA influence
...
#END
```

---

## Step 2: LLM System Prompt

Use constant system prompt:

```
You are consolidating macro topic labels generated in separate batches.

Your task is to merge labels that clearly refer to the same underlying subject.

Rules:
1. Merge only when labels clearly represent the same entity or domain.
2. Do not over-merge distinct subjects.
3. Keep the most stable, reusable label as canonical.
4. Every number must appear exactly once.

Output format:

canonical_label: n1,n2
canonical_label: n3
...

Only output canonical_label: numbers.
No explanations.
```

---

## Step 3: Parse and Validate

Validation rules:

* All macro numbers appear exactly once
* No duplicates
* No missing

Retry up to 3 times if invalid.

---

## Step 4: Build Final Mapping

You now build:

```
batch_macro_label → global_macro_label
```

Then:

```
canonical_topic → global_macro_label
```

This is your final macro mapping.

---

# 🔷 MODULE STRUCTURE

Implement the following:

### 1️⃣ `chunk_topics(topics: List[str], chunk_size=200) -> List[List[str]]`

### 2️⃣ `build_phase1_prompt(batch_topics, video_title, video_description) -> str`

### 3️⃣ `call_llm(system_prompt, user_prompt) -> str`

### 4️⃣ `parse_phase1_output(output, num_topics) -> Dict[macro_label, List[int]]`

### 5️⃣ `validate_number_coverage(mapping, expected_count) -> bool`

### 6️⃣ `build_phase2_prompt(macro_labels) -> str`

### 7️⃣ `parse_phase2_output(output, num_macros)`

### 8️⃣ `build_final_macro_mapping(...)`

### 9️⃣ Orchestrator:

`run_macro_labeller(topics, title, description) -> Dict[canonical_topic, global_macro]`

---

# 🔷 ADDITIONAL REQUIREMENTS

* Log all raw LLM responses
* Log validation failures
* Track number of retries
* Add timeout handling
* Fail gracefully

---

# 🔷 PERFORMANCE CONSIDERATIONS

* Use async batch calls if API allows
* Parallelize Phase 1 batches
* Phase 2 likely single call

---

# 🔷 OPTIONAL (BUT RECOMMENDED)

Cache macro mappings:

If the same canonical topic appears in another video:

* Try embedding similarity against known macro labels
* Auto-map if similarity > threshold
* Skip LLM call

---

# 🔷 FINAL OUTPUT STRUCTURE

Return:

```
{
  "canonical_to_macro": {...},
  "macro_to_canonicals": {...},
  "global_macro_list": [...]
}
```

---

# END GOAL

After running:

```
run_macro_labeller(...)
```

You should reduce ~3600 canonical topics to ~40–120 stable macro topics.

System must be:

* Reliable
* Deterministic
* Token-efficient
* Extensible

---