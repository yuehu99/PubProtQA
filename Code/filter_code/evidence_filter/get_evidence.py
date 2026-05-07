"""For every QA pair in grouped_by_protein_*.json that has a known PubMed_ID,
look up the corresponding abstract in abstracts.jsonl and ask an LLM (via
OpenRouter) to identify which sentences in the abstract support the answer
AND give a 0.0-1.0 score for how well the answer is supported.

To guarantee the saved evidence is verbatim, we pre-split each abstract into
numbered sentences (tracking byte offsets), send the numbered list to the
model, and ask it to return ONLY a JSON object with sentence indices and a
support score. We then slice the original abstract by the recorded offsets to
produce the evidence strings — so abstract.find(evidence) always succeeds.

Evidence + score are written into the grouped JSON under each conversation as
    "evidence":       ["sentence 1", "sentence 2", ...]
    "evidence_score": 0.85

The script is resumable: every API result is appended to a JSONL checkpoint as
soon as it returns, and on restart already-processed (uid, qa_idx) pairs are
skipped. After processing finishes, the checkpoint is merged into the grouped
JSON in a single rewrite.

Multi-key parallel:
    Pass one or more API keys via --keys (comma-separated) or via the env var
    LLM_API_KEYS (also comma-separated). A separate AsyncOpenAI client is
    created per key; workers round-robin across them so total throughput
    ≈ N × per-key concurrency.

Usage:
    python get_evidence.py --input <grouped.json> --abstracts <abstracts.jsonl> \\
                           --checkpoint <ckpt.jsonl> --keys KEY1,KEY2
    # or:
    LLM_API_KEYS=KEY1,KEY2 python get_evidence.py --input ... --abstracts ...
"""

import argparse
import asyncio
import itertools
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI, APIStatusError
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL    = "deepseek/deepseek-chat"
REQUEST_TIMEOUT  = 180   # seconds — hard cap so one slow call can't hang a worker

SYSTEM_PROMPT = """You are an evidence selector and grader for a biomedical QA dataset.

You will be given:
- a numbered list of sentences extracted from a research-paper abstract,
- a (Question, Answer) pair about a protein discussed in that abstract.

Tasks:
1. Identify which sentences from the list directly support the Answer.
2. Give a single support score in [0.0, 1.0] for how well the Answer is
   supported by the selected sentences:
     1.0  = Answer fully and directly stated
     0.7-0.9 = mostly supported, minor inference required
     0.4-0.6 = partially supported / requires non-trivial inference
     0.1-0.3 = weak / tangential support only
     0.0  = no support; selected list is empty or irrelevant

Rules:
- evidence_idx must be integer indices from the list, no paraphrases.
- If no sentence in the list supports the answer, return [] and score 0.0.
- Do not invent indices.
- Output JSON ONLY, no commentary, in this exact schema:
  {"evidence_idx": [3, 5], "score": 0.85}
"""

# Tokens that end with "." but should NOT terminate a sentence.
ABBREVIATIONS = {
    "e.g.", "i.e.", "et al.", "vs.", "cf.", "fig.", "figs.",
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.",
    "inc.", "ltd.", "co.", "no.", "nos.",
    "etc.", "approx.", "ca.", "pp.", "eq.", "ref.", "refs.",
    "sec.", "vol.", "vols.", "wt.", "pg.", "u.s.", "u.k.",
    "min.", "max.", "avg.", "std.", "resp.",
}

_BOUNDARY = re.compile(r"([.!?])(\s+)(?=[A-Z(\[0-9])")


def split_sentences_with_spans(text):
    """Split text into sentences keeping byte offsets.

    Returns list of (sentence_text, start, end) such that text[start:end] ==
    sentence_text. Section labels separated by '\\n' (e.g. "BACKGROUND:\\n...")
    are also handled because '\\s+' in the boundary regex matches newlines.
    """
    sentences = []
    cursor = 0
    pos = 0
    n = len(text)
    while pos < n:
        m = _BOUNDARY.search(text, pos)
        if not m:
            break
        end_punct = m.start() + 1
        tok_start = end_punct - 1
        while tok_start > cursor and not text[tok_start - 1].isspace():
            tok_start -= 1
        last_token = text[tok_start:end_punct].lower().lstrip("([{\"'")
        if last_token in ABBREVIATIONS:
            pos = m.end()
            continue
        s_start = cursor
        while s_start < end_punct and text[s_start].isspace():
            s_start += 1
        if s_start < end_punct:
            sentences.append((text[s_start:end_punct], s_start, end_punct))
        cursor = m.end()
        pos = m.end()

    if cursor < n:
        s_start = cursor
        while s_start < n and text[s_start].isspace():
            s_start += 1
        s_end = n
        while s_end > s_start and text[s_end - 1].isspace():
            s_end -= 1
        if s_start < s_end:
            sentences.append((text[s_start:s_end], s_start, s_end))
    return sentences


def load_abstracts(path):
    """Return {pmid: abstract_text} skipping empty / not_found entries."""
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(rec.get("pmid", "")).strip()
            ab = (rec.get("abstract") or "").strip()
            if pmid and ab:
                out[pmid] = ab
    return out


def load_checkpoint(path):
    """Return {(uid, qa_idx): {"evidence": [...], "score": float|None}}."""
    done = {}
    if not Path(path).exists():
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("uniprot_id")
            idx = rec.get("qa_idx")
            if uid is None or idx is None:
                continue
            done[(uid, int(idx))] = {
                "evidence": rec.get("evidence", []),
                "score":    rec.get("score"),
            }
    return done


def collect_tasks(grouped, abstracts, done):
    """Return tasks for QA pairs needing evidence + skip counters."""
    tasks = []
    skipped_unknown = 0
    skipped_no_abstract = 0
    for uid, entry in grouped.items():
        names = entry.get("protein_names", [])
        for idx, conv in enumerate(entry.get("conversations", [])):
            if (uid, idx) in done:
                continue
            # Skip if the output JSON already has evidence attached. The
            # "evidence" key being present (even if empty list) means the model
            # was already asked. evidence_score may not be set on legacy rows.
            if "evidence" in conv:
                done[(uid, idx)] = {
                    "evidence": conv.get("evidence") or [],
                    "score":    conv.get("evidence_score"),
                }
                continue
            pmid = str(conv.get("pubmed_id", "")).strip()
            if not pmid or pmid.lower() == "unknown":
                skipped_unknown += 1
                continue
            ab = abstracts.get(pmid)
            if not ab:
                skipped_no_abstract += 1
                continue
            sentences = split_sentences_with_spans(ab)
            if not sentences:
                skipped_no_abstract += 1
                continue
            tasks.append({
                "uniprot_id": uid,
                "qa_idx": idx,
                "pubmed_id": pmid,
                "protein_names": names,
                "question": conv.get("question", ""),
                "answer": conv.get("answer", ""),
                "abstract": ab,
                "sentences": sentences,
            })
    return tasks, skipped_unknown, skipped_no_abstract


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text, n_sentences):
    """Pull (evidence_idx, score) out of the model's JSON response.

    score is clamped to [0, 1]; missing / invalid -> None (caller can default).
    """
    if not text:
        return [], None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if obj is None:
        return [], None
    raw = obj.get("evidence_idx", obj.get("evidence", []))
    if isinstance(raw, (int, str)):
        raw = [raw]
    indices = []
    for x in raw:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_sentences and i not in indices:
            indices.append(i)

    score = obj.get("score")
    try:
        if score is not None:
            score = float(score)
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0
    except (TypeError, ValueError):
        score = None
    return indices, score


def format_sentences(sentences):
    return "\n".join(f"[{i}] {s}" for i, (s, _, _) in enumerate(sentences))


async def run(args):
    print(f"[info] loading abstracts from {args.abstracts}")
    abstracts = load_abstracts(args.abstracts)
    print(f"[info] abstracts available: {len(abstracts)}")

    print(f"[info] loading grouped JSON from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        grouped = json.load(f)
    print(f"[info] uniprot_ids: {len(grouped)}")

    done = load_checkpoint(args.checkpoint)
    print(f"[info] checkpoint contains: {len(done)} prior results")

    tasks, skip_unk, skip_noab = collect_tasks(grouped, abstracts, done)
    print(f"[info] QA pairs to process: {len(tasks)}")
    print(f"[info] skipped (Unknown PMID):     {skip_unk}")
    print(f"[info] skipped (no abstract found): {skip_noab}")

    keys = [k.strip() for k in (args.keys.split(",") if args.keys else []) if k.strip()]
    if not keys:
        raise SystemExit("no API keys provided (use --keys or LLM_API_KEYS env var)")
    print(f"[info] keys available for parallel use: {len(keys)}")

    if tasks:
        # One client per key — they don't share connection pools, so each gets
        # its own per-key rate budget.
        clients = [
            AsyncOpenAI(api_key=k, base_url=args.base_url, timeout=REQUEST_TIMEOUT)
            for k in keys
        ]
        # Round-robin counter for assigning a client to each call. asyncio is
        # single-threaded so a plain itertools.cycle is enough.
        rr = itertools.cycle(range(len(clients)))

        ckpt_lock = asyncio.Lock()
        ckpt_fh = open(args.checkpoint, "a", encoding="utf-8")
        progress = tqdm(total=len(tasks), desc="evidence", unit="qa")

        @retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(3))
        async def call_one(task, client_idx):
            client = clients[client_idx]
            user_msg = (
                f"Protein name(s): {', '.join(task['protein_names']) or task['uniprot_id']}\n"
                f"Sentences:\n{format_sentences(task['sentences'])}\n\n"
                f"Question: {task['question']}\n"
                f"Answer: {task['answer']}"
            )
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=2000,
                stream=False,
                timeout=REQUEST_TIMEOUT,
            )
            content = resp.choices[0].message.content or ""
            indices, score = parse_response(content, len(task["sentences"]))
            ab = task["abstract"]
            evidence = [ab[s:e] for i in indices for _, s, e in [task["sentences"][i]]]
            if not indices:
                # If the parser couldn't find indices and the model didn't give
                # a score, default score to 0.0 (no evidence).
                if score is None:
                    score = 0.0
            return indices, evidence, score, client_idx

        async def handle(task):
            client_idx = next(rr)
            tried_keys = []
            error = None
            indices, evidence, score = [], [], None
            # Try each key once on hard failures (e.g. one key out of credits
            # or rate-limited). Stops as soon as one succeeds.
            for _attempt in range(len(clients)):
                tried_keys.append(client_idx)
                try:
                    indices, evidence, score, client_idx = await call_one(task, client_idx)
                    error = None
                    break
                except Exception as e:
                    inner = e
                    if isinstance(e, RetryError):
                        try:
                            inner = e.last_attempt.exception() or e
                        except Exception:
                            inner = e
                    if isinstance(inner, APIStatusError):
                        body = ""
                        try:
                            body = (inner.response.text or "")[:300]
                        except Exception:
                            pass
                        error = f"APIStatusError {inner.status_code} key#{client_idx}: {body}"
                    else:
                        error = f"{type(inner).__name__} key#{client_idx}: {inner}"
                    client_idx = next(rr)  # rotate to a different key

            rec = {
                "uniprot_id":   task["uniprot_id"],
                "qa_idx":       task["qa_idx"],
                "pubmed_id":    task["pubmed_id"],
                "evidence_idx": indices,
                "evidence":     evidence,
                "score":        score,
                "key_idx":      tried_keys[-1] if tried_keys else None,
            }
            if error:
                rec["error"] = error
            async with ckpt_lock:
                ckpt_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ckpt_fh.flush()
            done[(task["uniprot_id"], task["qa_idx"])] = {
                "evidence": evidence,
                "score":    score,
            }
            progress.update(1)

        # Worker pool: bounded queue keeps memory flat and lets the first
        # workers start hitting the API immediately instead of waiting for the
        # event loop to schedule hundreds of thousands of pending coroutines.
        queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)

        async def worker():
            while True:
                t = await queue.get()
                if t is None:
                    queue.task_done()
                    return
                try:
                    await handle(t)
                finally:
                    queue.task_done()

        async def producer():
            for t in tasks:
                await queue.put(t)
            for _ in range(args.concurrency):
                await queue.put(None)

        workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        await producer()
        await asyncio.gather(*workers)

        progress.close()
        ckpt_fh.close()

    # ---- merge evidence + score back into grouped JSON ----
    n_attached = 0
    for (uid, idx), payload in done.items():
        entry = grouped.get(uid)
        if not entry:
            continue
        convs = entry.get("conversations", [])
        if 0 <= idx < len(convs):
            convs[idx]["evidence"] = payload["evidence"]
            convs[idx]["evidence_score"] = payload.get("score")
            n_attached += 1

    out_path = args.output or args.input
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    print(f"[done] evidence+score attached to {n_attached} QA pairs")
    print(f"[done] grouped JSON written to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True,
                   help="grouped-by-protein JSON (one record per protein with conversations)")
    p.add_argument("--abstracts",  required=True,
                   help="abstracts jsonl: one line per pmid with the abstract text")
    p.add_argument("--checkpoint", required=True,
                   help="incremental jsonl checkpoint (resumable)")
    p.add_argument("--output",     default=None,
                   help="path to write the augmented JSON; default = overwrite --input")
    p.add_argument("--concurrency", type=int, default=40,
                   help="total in-flight requests across all keys")
    p.add_argument("--model",      default=DEFAULT_MODEL,
                   help="LLM model id (default: deepseek/deepseek-chat)")
    p.add_argument("--keys",       default=os.environ.get("LLM_API_KEYS"),
                   help="comma-separated API keys (or set LLM_API_KEYS env var)")
    p.add_argument("--base-url",   default=DEFAULT_BASE_URL,
                   help="LLM API base URL (default: openrouter)")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
