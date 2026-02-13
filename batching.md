Good. This is the right moment to stop theorizing and design the **actual batching pipeline**.

You now have:

* ~6,000 unique topic keys (across signal types)
* Grouped as:
  `signal_type → topic_string → signals`

You need to:

> Turn these raw topic buckets into clean, LLM-ready batches
> for semantic consolidation + summarization.

Let’s design this cleanly and scalably.

---

# 🔵 Big Picture: What We’re Doing Next

You are NOT summarizing yet.

You are first:

1. Consolidating near-duplicate topic strings
2. Reducing 6,000 surface topics into ~500–1,000 canonical topics
3. Preparing LLM batches that are:

   * Small
   * Coherent
   * High-confidence

So the next stage is:

```
topic_strings
   ↓
embedding
   ↓
similarity grouping
   ↓
LLM consolidation batches
```

We are batching for **topic consolidation**, not insight generation yet.

---

# 🟢 Step 1 — Flatten Topics Across Signal Types

Even though you’re using signal_type-first for navigation, topic equivalence must ignore signal type.

Example:

* critique → "CIA influence"
* idea → "CIA influence"

These should merge into one canonical topic.

So first:

```
ALL_TOPICS = union of topic strings across all signal_types
```

You already have ~5986 unique keys.

Good.

---

# 🟢 Step 2 — Embed All Topic Strings

Generate embedding for each topic string.

Important:

* Normalize vectors
* Store index mapping: topic_id → vector

Now you have 5986 vectors.

---

# 🟢 Step 3 — Build Nearest Neighbor Index (FAISS)

Use FAISS to allow fast similarity search.

No clustering yet.

Just build index.

---

# 🟢 Step 4 — Create Candidate Topic Groups (Pre-LLM)

Now we need to create small groups of “likely equivalent” topics.

### Algorithm (Safe Version)

For each topic:

1. Retrieve top 20–30 nearest neighbors
2. Filter neighbors with similarity ≥ 0.88–0.90
3. Keep only **mutual neighbors**
4. Build small candidate groups
5. Mark visited topics so you don’t duplicate

This produces groups like:

```
Group 1:
- CIA influence
- CIA involvement
- CIA interference

Group 2:
- US relations
- US foreign relations
- India-US diplomacy

Singleton:
- novelty
```

You will likely reduce:

5986 topics → maybe 1000–1500 candidate groups.

That’s your first reduction layer.

---

# 🔴 Important: Control Group Size

Before sending to LLM, enforce:

* Max group size = 20–30 topics
* If group > 30:

  * Split by highest similarity to seed
  * Or split by similarity percentile

You do NOT want to send 200 topics to LLM.

Small batches only.

---

# 🟢 Step 5 — Build LLM Consolidation Batches

Each candidate group becomes 1 LLM batch.

Batch format:

```
You are consolidating topic labels.

Group these phrases into semantically equivalent clusters.
Split if necessary.
Provide a canonical label for each cluster.

Topic phrases:
1. CIA influence
2. CIA involvement
3. CIA interference
4. CIA propaganda
...
```

If group size = 1 → skip LLM.

---

# 🟢 Step 6 — LLM Output → Canonical Topic Map

You get:

```
Canonical topic: CIA involvement
Members:
- CIA influence
- CIA involvement
- CIA interference
- CIA propaganda
```

Store mapping:

```
original_topic → canonical_topic
```

Do this for all batches.

Now you have a reduced topic vocabulary.

---

# 🟢 Step 7 — Reassign Signals

Now update:

```
signal_type → canonical_topic → signals
```

This collapses fragmentation.

Now:

```
critique:
    CIA involvement → 14 signals
    US relations → 22 signals
```

Instead of 6 tiny variants.

---

# 🔵 How Many LLM Batches Will This Be?

Let’s estimate.

You have 5986 topics.

After similarity grouping:

* Assume 40% merge
* Leaves ~3500 candidate groups
* Of those:

  * Many are singletons (no LLM call)
  * Maybe 800–1200 groups size ≥ 2

So:
~1000 LLM consolidation calls.

Each small.
Each cheap.

Totally manageable.

---

# 🔴 DO NOT Do This

* ❌ Do not send all 6000 topics to LLM at once
* ❌ Do not use k-means
* ❌ Do not use HDBSCAN alone
* ❌ Do not merge across signal types separately
* ❌ Do not lower similarity threshold to 0.8

---

# 🟢 Now About Signal Type Separation

Important clarification:

Topic consolidation should ignore signal_type.

But later grouping remains:

```
signal_type → canonical_topic
```

So UI stays signal_type-first.

Architecture stays flexible.

---

# 🔵 After Topic Consolidation

Only after this do we move to:

* Topic-level summarization
* Batching signals per canonical topic
* Insight generation

Don’t mix these stages.

---

# 🟣 What You Should Build Now (Concrete Tasks)

1. Write embedding + FAISS neighbor retrieval
2. Implement high-threshold mutual similarity graph
3. Extract candidate groups
4. Cap group size
5. Prepare LLM batch prompt
6. Build canonical topic mapping

Do NOT jump ahead.

---

# 🧠 Key Design Principle

Right now you're building:

> A topic deduplication engine.

Not:

* A summarization engine
* A reasoning engine
* A clustering research project

Keep the step narrow and controlled.

---
