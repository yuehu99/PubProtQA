"""Stage 1 of the GO-consistency pipeline.

For every (uniprot_id, question, answer) in the raw test JSONL, ask the LLM to:
  1. decompose the answer into atomic claims,
  2. classify each claim by type,
  3. for GO-mappable types (BP / MF / CC) propose a GO term id + name.

We don't filter or verify here — we just produce a structured corpus that the
downstream verification step will compare against UniProt GO annotations.

Output (one line per QA, deduplicated across the raw file):
    {
      "uniprot_id": "P12345",
      "protein_names": ["..."],
      "question": "...",
      "answer": "...",
      "claims": [
        {"claim": "...", "type": "biological_process",
         "go_id": "GO:0006915", "go_name": "apoptotic process"},
        {"claim": "...", "type": "tissue_or_organ", "go_id": null, "go_name": null}
      ]
    }

The script is resumable — completed (uniprot_id, qa_hash) keys are skipped.

Usage:
    python extract_go_claims.py --input <raw QA jsonl> --output <claims jsonl> \\
                                --api-key <KEY>
    # or set the API key via environment variable:
    LLM_API_KEY=<KEY> python extract_go_claims.py --input ... --output ...
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI, APIStatusError
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL    = "deepseek-chat"
REQUEST_TIMEOUT  = 120

SYSTEM_PROMPT = """You are a biomedical claim extractor for QA evaluation.

Given a Question and an Answer about a protein, decompose the Answer into
atomic, single-fact claims. For each claim assign a type and, when applicable,
the most likely Gene Ontology (GO) **term name** (NOT id).

Claim types:
- biological_process : a biological process / pathway the protein takes part in        (maps to GO BP)
- molecular_function : a biochemical activity such as binding, catalysis, transport    (maps to GO MF)
- cellular_component : a subcellular location such as nucleus, membrane, ribosome      (maps to GO CC)
- protein_identity   : names, family / superfamily membership, isoforms, EC number
- tissue_or_organ    : tissue, organ, anatomical structure, or cell type
- disease            : disease, phenotype, or clinical condition
- organism           : species, strain, or taxonomic group
- interaction        : interacts / binds with a NAMED partner molecule
- experimental       : an experimental method, condition, or measurement
- other              : anything that does not fit the above

GO name rules:
- Fill go_name ONLY when the claim type is biological_process,
  molecular_function, or cellular_component.
- Use the CANONICAL GO term name as it appears in the Gene Ontology
  (e.g. "apoptotic process" not "apoptosis"; "negative regulation of
  growth hormone secretion" not "inhibits GH release").
- If you are not confident which exact GO term matches, give the closest
  canonical term name you can — partial matches are still useful for lookup.
  If you have no idea, set go_name to null.
- Do NOT output GO ids. We resolve ids from the name in a separate step.

Other rules:
- Each claim must be a single fact (≤25 words, no "and"/"or" joining unrelated facts).
- Stay grounded in the Answer text — do not introduce facts that aren't there.
- If the Answer contains no extractable claim, return {"claims": []}.

Return JSON ONLY, no commentary, in this exact schema:
{"claims": [
  {"claim": "...", "type": "...", "go_name": "..." | null}
]}
"""


def qa_hash(question, answer):
    h = hashlib.sha1()
    h.update(question.encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(answer.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def iter_qa(input_path):
    """Yield (uniprot_id, protein_names, question, answer) deduplicated."""
    seen = set()
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for prot in rec.get("proteins", []):
                uid = prot.get("uniprot_id")
                if not uid:
                    continue
                name = prot.get("protein") or uid
                for conv in prot.get("conversations", []):
                    q = conv.get("Question") or ""
                    a = conv.get("Answer") or ""
                    if not q or not a:
                        continue
                    key = (uid, qa_hash(q, a))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield uid, name, q, a


def load_done(path):
    """Load (uid, qa_hash) keys for every QA already in the output file so we
    can skip them on resume. Falls back to recomputing qa_hash from the stored
    question+answer when the record predates the qa_hash field."""
    done = set()
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
            if not uid:
                continue
            qh = rec.get("qa_hash")
            if not qh:
                q = rec.get("question") or ""
                a = rec.get("answer") or ""
                if not q or not a:
                    continue
                qh = qa_hash(q, a)
            done.add((uid, qh))
    return done


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_TYPES = {
    "biological_process", "molecular_function", "cellular_component",
    "protein_identity", "tissue_or_organ", "disease", "organism",
    "interaction", "experimental", "other",
}
_GO_TYPES = {"biological_process", "molecular_function", "cellular_component"}


def parse_claims(text):
    if not text:
        return []
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
    if not isinstance(obj, dict):
        return []
    raw = obj.get("claims", [])
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        claim = (c.get("claim") or "").strip()
        ctype = (c.get("type") or "other").strip().lower()
        if ctype not in _VALID_TYPES:
            ctype = "other"
        go_name = c.get("go_name")
        if ctype not in _GO_TYPES:
            go_name = None
        else:
            if isinstance(go_name, str) and go_name.strip():
                go_name = go_name.strip()
            else:
                go_name = None
        if not claim:
            continue
        # Note: we deliberately do NOT keep any LLM-supplied go_id — the LLM is
        # not reliable at producing real GO ids. Resolution happens in the
        # separate stage 1.5 script (resolve_go_names.py).
        out.append({"claim": claim, "type": ctype, "go_name": go_name})
    return out


async def run(args):
    print(f"[info] reading raw QA from {args.input}")
    qa_iter = iter_qa(args.input)
    qa_list = list(qa_iter)
    print(f"[info] unique (uid, qa) pairs: {len(qa_list)}")

    done = load_done(args.output)
    print(f"[info] checkpoint already has: {len(done)}")

    todo = [(uid, name, q, a) for (uid, name, q, a) in qa_list
            if (uid, qa_hash(q, a)) not in done]
    print(f"[info] to process: {len(todo)}")
    if not todo:
        print("[info] nothing to do.")
        return

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url,
                         timeout=REQUEST_TIMEOUT)
    ckpt_lock = asyncio.Lock()
    out_fh = open(args.output, "a", encoding="utf-8")
    progress = tqdm(total=len(todo), desc="extract", unit="qa")

    @retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(3))
    async def call_one(uid, name, q, a):
        user_msg = (
            f"Protein: {name}\n"
            f"Question: {q}\n"
            f"Answer: {a}"
        )
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=800,
            temperature=0.0,
            stream=False,
            timeout=REQUEST_TIMEOUT,
        )
        return resp.choices[0].message.content or ""

    async def handle(uid, name, q, a):
        try:
            content = await call_one(uid, name, q, a)
            claims = parse_claims(content)
            error = None
        except Exception as e:
            inner = e
            if isinstance(e, RetryError):
                try:
                    inner = e.last_attempt.exception() or e
                except Exception:
                    pass
            if isinstance(inner, APIStatusError):
                body = ""
                try:
                    body = (inner.response.text or "")[:300]
                except Exception:
                    pass
                error = f"APIStatusError {inner.status_code}: {body}"
            else:
                error = f"{type(inner).__name__}: {inner}"
            claims = []
        rec = {
            "uniprot_id":    uid,
            "protein_name":  name,
            "qa_hash":       qa_hash(q, a),
            "question":      q,
            "answer":        a,
            "claims":        claims,
        }
        if error:
            rec["error"] = error
        async with ckpt_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
        progress.update(1)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)

    async def worker():
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            try:
                await handle(*item)
            finally:
                queue.task_done()

    async def producer():
        for item in todo:
            await queue.put(item)
        for _ in range(args.concurrency):
            await queue.put(None)

    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    await producer()
    await asyncio.gather(*workers)

    progress.close()
    out_fh.close()
    print(f"[done] checkpoint: {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True,
                   help="raw QA jsonl (one record per paper-protein group)")
    p.add_argument("--output",      required=True,
                   help="path to write extracted-claims jsonl (resumable)")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help="LLM model name (default: deepseek-chat)")
    p.add_argument("--api-key",     default=os.environ.get("LLM_API_KEY"),
                   help="LLM API key (or set LLM_API_KEY env variable)")
    p.add_argument("--base-url",    default=DEFAULT_BASE_URL,
                   help="LLM API base URL (default: deepseek)")
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()
    if not args.api_key:
        p.error("--api-key (or LLM_API_KEY env var) is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
