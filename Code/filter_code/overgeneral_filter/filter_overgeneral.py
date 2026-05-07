"""Flag potentially over-generalized answers in pubprotQA.jsonl.

For each (uniprot_id, qa_idx) we compute three independent checks. The final
`overgeneral_score` is the sum (0..3, higher = more suspicious).

Check 1 — DANGER WORDS  (deterministic, cheap)
    Hit if the answer contains any of:
      • absolutist:    only / always / never / completely / entirely / exclusively
      • strong causal: controls / determines / drives / enables / is responsible for
      • over-certain:  is the key / is essential / is critical

Check 2 — HEDGING MISMATCH  (deterministic, cheap)
    Hit if the EVIDENCE sentences contain a hedging word
      (may / might / suggests / likely / appears / primarily)
    but the ANSWER contains NONE of those.
    → evidence is uncertain, model removed the uncertainty.
    Skipped (no hit) when evidence is missing — we can't compare.

Check 3 — ASSOCIATION → CAUSATION  (LLM judge, optional)
    LLM (5-key OpenRouter pool, round-robin) judges whether the answer
    overstates causation/mechanism beyond what the evidence supports.

Output JSONL — same schema as pubprotQA.jsonl but each conversation gains:
    overgeneral_score          : 0|1|2|3
    overgeneral_danger_hits    : ["only", "is essential", ...]
    overgeneral_hedging_mismatch : true | false
    overgeneral_causation      : "yes" | "no" | null   (null = check 3 skipped)
    overgeneral_causation_reason : "..." | null

Usage
-----
    # cheap pass — checks 1 + 2 only (very fast, handles 800K conv in ~minutes)
    python filter_overgeneral.py

    # also run check 3 (LLM) on EVERY conversation
    python filter_overgeneral.py --llm

    # only run check 3 on already-suspicious conversations (score from 1+2 >= 1)
    python filter_overgeneral.py --llm --llm-only-suspicious

    # resume — already-scored convs in the output are skipped
    python filter_overgeneral.py --llm --resume

    # smaller scope (e.g. high-evidence subset only)
    python filter_overgeneral.py --input <high_evidence.jsonl> \\
                                 --output <high_overgeneral.jsonl>
"""

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Word lists (Check 1 + Check 2)
# ---------------------------------------------------------------------------

DANGER_ABSOLUTE   = ["only", "always", "never", "completely", "entirely", "exclusively"]
DANGER_CAUSAL     = ["controls", "determines", "drives", "enables"]
DANGER_OVERCERTAIN_PHRASES = [
    "is the key", "is essential", "is critical", "is responsible for",
]
HEDGING = ["may", "might", "suggests", "suggest", "suggested",
           "likely", "appears", "appear", "primarily"]


def _word_re(words):
    """Compile case-insensitive whole-word regex matching any term in `words`."""
    pat = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
    return re.compile(pat, flags=re.IGNORECASE)


def _phrase_re(phrases):
    """Compile case-insensitive multi-word phrase regex (no \\b at edges between
    words because phrases already include spaces)."""
    pat = "|".join(re.escape(p) for p in phrases)
    return re.compile(pat, flags=re.IGNORECASE)


_DANGER_WORDS_RE   = _word_re(DANGER_ABSOLUTE + DANGER_CAUSAL)
_DANGER_PHRASES_RE = _phrase_re(DANGER_OVERCERTAIN_PHRASES)
_HEDGING_RE        = _word_re(HEDGING)


# ---------------------------------------------------------------------------
# Check 1 + Check 2
# ---------------------------------------------------------------------------

def check1_danger_hits(answer):
    """Return list of distinct danger terms found in the answer (lowercased)."""
    if not answer:
        return []
    found = []
    seen = set()
    for m in _DANGER_WORDS_RE.finditer(answer):
        w = m.group(1).lower()
        if w not in seen:
            seen.add(w)
            found.append(w)
    for m in _DANGER_PHRASES_RE.finditer(answer):
        p = m.group(0).lower()
        if p not in seen:
            seen.add(p)
            found.append(p)
    return found


def check2_hedging_mismatch(answer, evidence):
    """True iff evidence has a hedging word but the answer has none.

    `evidence` may be a list of strings or a single string. If evidence is
    empty we return False (we don't penalize on missing evidence)."""
    if not evidence:
        return False
    if isinstance(evidence, list):
        ev_text = " ".join(evidence)
    else:
        ev_text = str(evidence)
    if not ev_text.strip():
        return False
    ev_has = _HEDGING_RE.search(ev_text) is not None
    if not ev_has:
        return False
    ans_has = _HEDGING_RE.search(answer or "") is not None
    return not ans_has


# ---------------------------------------------------------------------------
# Check 3 — LLM (optional)
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL    = "deepseek/deepseek-chat"
REQUEST_TIMEOUT  = 90

LLM_SYSTEM_PROMPT = """You are a scientific writing reviewer for biomedical QA.

You will be shown:
- a Question about a protein
- an Answer
- (optionally) the EVIDENCE sentences from the underlying paper abstract

Decide whether the Answer overstates a causal or mechanistic claim that the
evidence does not actually support — for example, presenting a correlation /
association as a definitive causal effect, or stating an outcome as certain
when the evidence only suggests / implies it.

Reply with ONLY a JSON object, no commentary, in this exact schema:
  {"verdict": "yes" | "no", "reason": "<= 25 words"}

verdict = "yes" : the answer overstates causation/certainty beyond the evidence
verdict = "no"  : the answer is appropriately calibrated
"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_verdict(text):
    if not text:
        return None, ""
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
    obj = obj or {}
    v = (obj.get("verdict") or "").strip().lower()
    if v not in ("yes", "no"):
        v = None
    return v, (obj.get("reason") or "").strip()[:200]


# ---------------------------------------------------------------------------
# I/O + main
# ---------------------------------------------------------------------------

def iter_input_records(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[warn] line {line_no}: bad json ({e})", file=sys.stderr)


def load_done(path):
    """Return {(uid, qa_idx): rec_with_score} for already-processed conversations
    so we can resume without recomputing. We keep the full record so we can
    decide whether check 3 (LLM) needs to run for it."""
    done = {}
    if not Path(path).exists():
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            uid = rec.get("uniprot_id")
            for i, c in enumerate(rec.get("conversations") or []):
                if "overgeneral_score" in c:
                    done[(uid, i)] = c
    return done


def deterministic_pass(args):
    """One streaming pass through input. Run checks 1 + 2 for every QA, write
    the augmented record to output. If --resume and the QA was already
    scored we keep the existing values (so we don't lose a prior LLM verdict)."""
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output) if args.resume else {}
    print(f"[info] resume: {len(done)} prior conversations already scored")

    n_lines = 0
    n_conv = 0
    score_hist = Counter()
    danger_top = Counter()

    tmp = args.output + ".tmp"
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(tmp,         "w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            uid = rec.get("uniprot_id")
            convs = rec.get("conversations") or []
            new_convs = []
            for i, c in enumerate(convs):
                # Resume — keep the prior augmented version verbatim.
                if (uid, i) in done:
                    new_c = done[(uid, i)]
                else:
                    answer = c.get("answer", "") or ""
                    evidence = c.get("evidence")
                    danger = check1_danger_hits(answer)
                    hed_mismatch = check2_hedging_mismatch(answer, evidence)
                    score = (1 if danger else 0) + (1 if hed_mismatch else 0)
                    new_c = dict(c)
                    new_c["overgeneral_danger_hits"] = danger
                    new_c["overgeneral_hedging_mismatch"] = hed_mismatch
                    new_c["overgeneral_causation"] = None
                    new_c["overgeneral_causation_reason"] = None
                    new_c["overgeneral_score"] = score
                new_convs.append(new_c)
                n_conv += 1
                score_hist[new_c.get("overgeneral_score", 0)] += 1
                for w in new_c.get("overgeneral_danger_hits") or []:
                    danger_top[w] += 1
            rec["conversations"] = new_convs
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_lines += 1
            if n_lines % 50000 == 0:
                print(f"[progress] {n_lines:,} proteins scanned, {n_conv:,} convs")
    os.replace(tmp, args.output)

    print()
    print(f"=== Stage 1+2 (deterministic) done ===")
    print(f"  proteins:      {n_lines:,}")
    print(f"  conversations: {n_conv:,}")
    print(f"  score distribution (without LLM):")
    for s in (0, 1, 2):
        n = score_hist[s]
        pct = 100 * n / max(1, n_conv)
        print(f"    score={s}: {n:>9,}  ({pct:5.2f}%)")
    print(f"  top danger-word hits:")
    for w, n in danger_top.most_common(15):
        print(f"    {w:<25} {n:>8,}")
    print(f"\n[info] wrote: {args.output}")
    return n_conv


async def llm_pass(args):
    """Augment each conversation with check 3 (LLM). Runs ONLY where
    overgeneral_causation is null. Updates output file in place."""
    from openai import AsyncOpenAI, APIStatusError
    from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError
    from tqdm import tqdm

    # Build full task list. Re-read the freshly-written output (which has
    # check 1+2 results) and decide which convs need LLM.
    print(f"[info] scanning {args.output} for convs needing LLM check 3 ...")
    tasks = []  # list of (uid, qa_idx, payload-for-LLM)
    n_total = 0
    with open(args.output, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            uid = rec.get("uniprot_id")
            for i, c in enumerate(rec.get("conversations") or []):
                n_total += 1
                if c.get("overgeneral_causation") is not None:
                    continue  # already done
                if args.llm_only_suspicious and (c.get("overgeneral_score") or 0) < 1:
                    continue
                evidence = c.get("evidence")
                if isinstance(evidence, list):
                    ev_text = " ".join(evidence)[:1500]
                else:
                    ev_text = (str(evidence) if evidence else "")[:1500]
                tasks.append((uid, i, {
                    "question": (c.get("question", "") or "")[:600],
                    "answer":   (c.get("answer", "") or "")[:1500],
                    "evidence": ev_text,
                }))
    print(f"[info]   {n_total:,} convs total, {len(tasks):,} need LLM check")
    if not tasks:
        return

    keys = [k.strip() for k in (args.keys.split(",") if args.keys else []) if k.strip()]
    if not keys:
        raise SystemExit("no API keys provided (use --keys or LLM_API_KEYS env var)")
    clients = [AsyncOpenAI(api_key=k, base_url=args.base_url, timeout=REQUEST_TIMEOUT) for k in keys]
    rr = itertools.cycle(range(len(clients)))

    @retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(3))
    async def call_one(client, payload):
        user = (
            f"Question: {payload['question']}\n"
            f"Answer: {payload['answer']}\n"
            f"Evidence: {payload['evidence']}"
        )
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user},
            ],
            max_tokens=120,
            temperature=0.0,
            stream=False,
            timeout=REQUEST_TIMEOUT,
        )
        return resp.choices[0].message.content or ""

    verdicts = {}  # (uid, idx) -> (verdict, reason)
    progress = tqdm(total=len(tasks), desc="llm", unit="qa")
    sem = asyncio.Semaphore(args.concurrency)

    async def handle(uid, idx, payload):
        async with sem:
            client_idx = next(rr)
            verdict, reason = None, ""
            for _ in range(len(clients)):
                try:
                    content = await call_one(clients[client_idx], payload)
                    verdict, reason = _parse_verdict(content)
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
                            body = (inner.response.text or "")[:200]
                        except Exception:
                            pass
                        reason = f"APIStatusError {inner.status_code}: {body}"[:200]
                    else:
                        reason = f"{type(inner).__name__}: {inner}"[:200]
                    client_idx = next(rr)
            verdicts[(uid, idx)] = (verdict, reason)
            progress.update(1)

            # Periodic snapshot to make sure long runs don't lose work.
            if len(verdicts) % args.snapshot_every == 0:
                _apply_verdicts(args.output, verdicts)
                verdicts.clear()

    await asyncio.gather(*(handle(u, i, p) for u, i, p in tasks))
    progress.close()
    if verdicts:
        _apply_verdicts(args.output, verdicts)
    print(f"[done] check 3 (LLM) finished")


def _apply_verdicts(path, verdicts):
    """Read JSONL, attach verdicts in `verdicts` to the matching conversations,
    update `overgeneral_score`, write back. Atomic rewrite via tmp file."""
    if not verdicts:
        return
    tmp = path + ".tmp"
    n_applied = 0
    with open(path, "r", encoding="utf-8") as fin, \
         open(tmp,  "w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            uid = rec.get("uniprot_id")
            for i, c in enumerate(rec.get("conversations") or []):
                key = (uid, i)
                if key in verdicts:
                    v, why = verdicts[key]
                    c["overgeneral_causation"] = v
                    c["overgeneral_causation_reason"] = why
                    base = (1 if c.get("overgeneral_danger_hits") else 0) + \
                           (1 if c.get("overgeneral_hedging_mismatch") else 0)
                    add  = 1 if v == "yes" else 0
                    c["overgeneral_score"] = base + add
                    n_applied += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[snapshot] applied {n_applied} new LLM verdicts → {path}")


def summarize(path):
    """Quick distribution summary of overgeneral_score across the output file."""
    if not Path(path).exists():
        print(f"[warn] no such file: {path}")
        return
    score_hist = Counter()
    danger_top = Counter()
    hedging_n = 0
    causation_n = 0
    causation_yes = 0
    n_conv = 0
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for c in rec.get("conversations") or []:
                n_conv += 1
                score_hist[c.get("overgeneral_score", 0)] += 1
                for w in c.get("overgeneral_danger_hits") or []:
                    danger_top[w] += 1
                if c.get("overgeneral_hedging_mismatch"):
                    hedging_n += 1
                v = c.get("overgeneral_causation")
                if v in ("yes", "no"):
                    causation_n += 1
                    if v == "yes":
                        causation_yes += 1
    print()
    print(f"=== summary ({path}) ===")
    print(f"  conversations: {n_conv:,}")
    for s in sorted(score_hist):
        n = score_hist[s]
        print(f"  score={s}: {n:>9,}  ({100*n/max(1,n_conv):5.2f}%)")
    print(f"\n  hedging mismatches: {hedging_n:,}  ({100*hedging_n/max(1,n_conv):.2f}%)")
    if causation_n:
        print(f"  LLM causation judged: {causation_n:,}  "
              f"yes={causation_yes:,} ({100*causation_yes/causation_n:.2f}% of judged)")
    else:
        print(f"  LLM causation: not run yet")
    print(f"  top danger-word hits:")
    for w, n in danger_top.most_common(15):
        print(f"    {w:<25} {n:>8,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True,
                    help="input jsonl with conversations to score")
    ap.add_argument("--output", required=True,
                    help="path to write augmented jsonl with overgeneral_score")
    ap.add_argument("--llm",    action="store_true",
                    help="also run check 3 (LLM) — expensive on full file")
    ap.add_argument("--llm-only-suspicious", action="store_true",
                    help="restrict check 3 to convs already flagged by check 1 or 2")
    ap.add_argument("--model",       default=DEFAULT_MODEL,
                    help="LLM model id (default: deepseek/deepseek-chat)")
    ap.add_argument("--keys",        default=os.environ.get("LLM_API_KEYS"),
                    help="comma-separated API keys (or set LLM_API_KEYS env var)")
    ap.add_argument("--base-url",    default=DEFAULT_BASE_URL,
                    help="LLM API base URL (default: openrouter)")
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--snapshot-every", type=int, default=2000,
                    help="apply LLM verdicts to disk every N completed calls")
    ap.add_argument("--resume",   action="store_true",
                    help="skip QAs that are already scored in --output")
    ap.add_argument("--summarize-only", action="store_true",
                    help="just print stats from an existing output file")
    args = ap.parse_args()

    if args.summarize_only:
        summarize(args.output)
        return

    t0 = time.time()
    deterministic_pass(args)
    if args.llm:
        asyncio.run(llm_pass(args))
    summarize(args.output)
    print(f"\n[total time] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
