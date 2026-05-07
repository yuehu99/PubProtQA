"""Stage 2 of the GO-consistency pipeline.

Inputs
------
- a resolved-claims jsonl (output of resolve_go_names.py)

Steps
-----
1. Collect every (uniprot_id, claim_go_id, claim_type) where the claim is a
   BP / MF / CC claim that the LLM successfully mapped to a GO id.
2. Fetch each protein's curated GO annotations from UniProt (batched, cached).
3. Optionally load go-basic.obo to enable ancestor / descendant matching.
4. For each (uid, claim_go_id), compute:
     exact_match     - claim_go_id ∈ UniProt(uid)[aspect]
     ancestor_match  - any ancestor of claim_go_id ∈ UniProt(uid)[aspect]
                       (claim is more general than something annotated)
     descendant_match- any descendant of claim_go_id ∈ UniProt(uid)[aspect]
                       (claim is more specific than something annotated)
   "consistent" = exact_match OR ancestor_match OR descendant_match.
5. Write per-claim verdicts and print aggregate metrics by aspect.

Usage
-----
    python verify_go_consistency.py --claims <resolved.jsonl> --output <out.jsonl> \\
                                    --mismatch <mismatch.jsonl> --obo <go-basic.obo> \\
                                    --uniprot-cache <cache.jsonl>
    python verify_go_consistency.py ... --aspects P
    python verify_go_consistency.py ... --skip-ancestor
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
GO_OBO_URL     = "http://purl.obolibrary.org/obo/go/go-basic.obo"

TYPE_TO_ASPECT = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}
ASPECT_NAME = {"P": "biological_process", "F": "molecular_function", "C": "cellular_component"}

_GO_ID_IN_TEXT = re.compile(r"\[(GO:\d{7})\]")
_GO_ID = re.compile(r"^GO:\d{7}$")


# -------------------- claim loading --------------------

def load_claims(path):
    """Yield {uniprot_id, claim, go_id, type, aspect, qa_hash} for GO claims with go_id."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            uid = rec.get("uniprot_id")
            if not uid:
                continue
            for c in rec.get("claims", []):
                ctype = c.get("type")
                aspect = TYPE_TO_ASPECT.get(ctype)
                if not aspect:
                    continue
                gid = c.get("go_id")
                if not (isinstance(gid, str) and _GO_ID.match(gid)):
                    continue
                yield {
                    "uniprot_id": uid,
                    "qa_hash":    rec.get("qa_hash"),
                    "claim":      c.get("claim", ""),
                    "go_id":      gid,
                    "go_name":    c.get("go_name"),
                    "type":       ctype,
                    "aspect":     aspect,
                }


# -------------------- UniProt GO fetching --------------------

def load_uniprot_cache(path):
    """Return {uid: {"P": set, "F": set, "C": set}} from prior runs."""
    cache = {}
    if not Path(path).exists():
        return cache
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
            if uid:
                cache[uid] = {a: set(rec.get(a, [])) for a in "PFC"}
    return cache


def http_get(url, params=None, retries=5):
    if params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(params)}"
    last = None
    for attempt in range(retries):
        try:
            req = Request(url)
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
            while data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data
        except HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise
        except URLError as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"http_get failed: {last}")


def fetch_uniprot_go(missing_ids, cache_path, batch_size=100):
    """Fetch GO terms for the given UniProt accessions and append to cache."""
    if not missing_ids:
        return {}
    fetched = {}
    out_fh = open(cache_path, "a", encoding="utf-8")
    ids = list(missing_ids)
    print(f"[info] fetching UniProt GO for {len(ids)} accessions in batches of {batch_size}")
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        query = " OR ".join(f"accession:{a}" for a in batch)
        try:
            raw = http_get(UNIPROT_STREAM, params={
                "query":  query,
                "format": "tsv",
                "fields": "accession,go_p,go_f,go_c",
            })
        except Exception as e:
            print(f"[warn] batch {i // batch_size}: {e}", file=sys.stderr)
            continue
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            continue
        seen_in_batch = set()
        # First line is header; ignore. Columns: Entry, Gene Ontology (biological process), ...
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            acc, gp, gf, gc = cols[0], cols[1], cols[2], cols[3]
            entry = {
                "uniprot_id": acc,
                "P": sorted(set(_GO_ID_IN_TEXT.findall(gp))),
                "F": sorted(set(_GO_ID_IN_TEXT.findall(gf))),
                "C": sorted(set(_GO_ID_IN_TEXT.findall(gc))),
            }
            fetched[acc] = {a: set(entry[a]) for a in "PFC"}
            seen_in_batch.add(acc)
            out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Record empties for accessions UniProt didn't return — they exist but
        # have no curated GO terms, or are obsolete. Avoid re-fetching next run.
        for acc in batch:
            if acc not in seen_in_batch:
                empty = {"uniprot_id": acc, "P": [], "F": [], "C": []}
                fetched[acc] = {a: set() for a in "PFC"}
                out_fh.write(json.dumps(empty, ensure_ascii=False) + "\n")
        out_fh.flush()
        print(f"[progress] {min(i + batch_size, len(ids))}/{len(ids)} accessions")
        time.sleep(0.2)
    out_fh.close()
    return fetched


# -------------------- GO ontology (ancestors / descendants) --------------------

def ensure_obo(path):
    if Path(path).exists():
        return
    print(f"[info] downloading go-basic.obo to {path}")
    data = http_get(GO_OBO_URL)
    with open(path, "wb") as f:
        f.write(data)


def load_go_graph(path):
    """Parse go-basic.obo into:
        is_a_parents[term]    = set of direct parents (via is_a / part_of)
        ancestors[term]       = set of all ancestors (transitive closure)
        descendants[term]     = set of all descendants
        alt_to_main           = {alt_id: main_id}
        aspect[term]          = "P" | "F" | "C"
    """
    parents = defaultdict(set)
    children = defaultdict(set)
    aspect = {}
    alt_to_main = {}
    cur = None
    cur_alts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("[Term]"):
                if cur:
                    for a in cur_alts:
                        alt_to_main[a] = cur["id"]
                cur = {"id": None, "is_a": [], "part_of": [], "ns": None, "obs": False}
                cur_alts = []
            elif line.startswith("[") and cur:
                # entered a non-Term stanza (e.g. [Typedef]); flush.
                for a in cur_alts:
                    alt_to_main[a] = cur["id"]
                cur = None
                cur_alts = []
            elif cur is not None:
                if line.startswith("id: "):
                    cur["id"] = line[4:].strip()
                elif line.startswith("alt_id: "):
                    cur_alts.append(line[8:].strip())
                elif line.startswith("namespace: "):
                    ns = line[11:].strip()
                    cur["ns"] = {"biological_process": "P",
                                 "molecular_function": "F",
                                 "cellular_component": "C"}.get(ns)
                elif line.startswith("is_a: "):
                    parent = line[6:].split("!")[0].strip()
                    cur["is_a"].append(parent)
                elif line.startswith("relationship: part_of "):
                    parent = line[len("relationship: part_of "):].split("!")[0].strip()
                    cur["part_of"].append(parent)
                elif line.startswith("is_obsolete: true"):
                    cur["obs"] = True
                elif line == "":
                    if cur.get("id") and not cur["obs"]:
                        if cur["ns"]:
                            aspect[cur["id"]] = cur["ns"]
                        for p in cur["is_a"] + cur["part_of"]:
                            parents[cur["id"]].add(p)
                            children[p].add(cur["id"])
                        for a in cur_alts:
                            alt_to_main[a] = cur["id"]
                    cur = None
                    cur_alts = []
    # Final flush.
    if cur and cur.get("id") and not cur["obs"]:
        if cur["ns"]:
            aspect[cur["id"]] = cur["ns"]
        for p in cur["is_a"] + cur["part_of"]:
            parents[cur["id"]].add(p)
            children[p].add(cur["id"])
        for a in cur_alts:
            alt_to_main[a] = cur["id"]

    # Transitive closure using DFS with memoization.
    ancestors = {}
    descendants = {}

    def get_ancestors(t):
        if t in ancestors:
            return ancestors[t]
        seen = set()
        stack = list(parents.get(t, ()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(parents.get(x, ()))
        ancestors[t] = seen
        return seen

    def get_descendants(t):
        if t in descendants:
            return descendants[t]
        seen = set()
        stack = list(children.get(t, ()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(children.get(x, ()))
        descendants[t] = seen
        return seen

    all_terms = set(parents) | set(children) | set(aspect)
    print(f"[info] go-basic.obo: {len(all_terms)} terms, {len(alt_to_main)} alt ids")
    return {
        "parents":    parents,
        "ancestors":  get_ancestors,
        "descendants": get_descendants,
        "alt_to_main": alt_to_main,
        "aspect":     aspect,
    }


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--claims",         required=True,
                   help="resolved-claims jsonl from resolve_go_names.py")
    p.add_argument("--uniprot-cache",  required=True,
                   help="path to a UniProt GO cache jsonl (created/extended on the fly)")
    p.add_argument("--obo",            required=True,
                   help="path to go-basic.obo (downloaded if missing)")
    p.add_argument("--output",         required=True,
                   help="path to write per-claim verdicts jsonl")
    p.add_argument("--mismatch",       required=True,
                   help="path to write claims that did not match (consistent=False)")
    p.add_argument("--aspects",        default="P",
                   help="comma-separated subset of P,F,C (default: P only — MVP)")
    p.add_argument("--skip-ancestor",  action="store_true",
                   help="exact match only; do not load go-basic.obo")
    args = p.parse_args()

    aspects_wanted = set(a.strip().upper() for a in args.aspects.split(","))
    if not aspects_wanted.issubset({"P", "F", "C"}):
        sys.exit(f"--aspects must be subset of P,F,C (got {aspects_wanted})")

    print(f"[info] loading claims from {args.claims}")
    claims = [c for c in load_claims(args.claims) if c["aspect"] in aspects_wanted]
    print(f"[info] GO claims with go_id: {len(claims)}")
    if not claims:
        sys.exit("nothing to verify")

    needed_uids = sorted({c["uniprot_id"] for c in claims})
    print(f"[info] unique proteins to look up: {len(needed_uids)}")

    cache = load_uniprot_cache(args.uniprot_cache)
    missing = [u for u in needed_uids if u not in cache]
    print(f"[info] cache hits: {len(needed_uids) - len(missing)}, misses: {len(missing)}")
    new = fetch_uniprot_go(missing, args.uniprot_cache)
    cache.update(new)

    if not args.skip_ancestor:
        ensure_obo(args.obo)
        graph = load_go_graph(args.obo)
        alt_to_main = graph["alt_to_main"]
        anc_fn = graph["ancestors"]
        desc_fn = graph["descendants"]
    else:
        graph = None
        alt_to_main = {}
        anc_fn = lambda t: set()
        desc_fn = lambda t: set()

    def canonical(go_id):
        return alt_to_main.get(go_id, go_id)

    # ---- per-claim verdicts ----
    Path(args.mismatch).parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(args.output, "w", encoding="utf-8")
    miss_fh = open(args.mismatch, "w", encoding="utf-8")
    n_mismatch = 0
    counts = defaultdict(lambda: {"n": 0, "exact": 0, "anc": 0, "desc": 0,
                                   "consistent": 0, "no_uniprot_terms": 0})
    for c in claims:
        uid = c["uniprot_id"]
        aspect = c["aspect"]
        gid = canonical(c["go_id"])
        uniprot_terms = cache.get(uid, {}).get(aspect, set())
        # canonicalise UniProt terms too (rare alt_ids)
        uniprot_canon = {canonical(t) for t in uniprot_terms}

        exact = gid in uniprot_canon
        ancs  = anc_fn(gid) if not exact else set()
        desc  = desc_fn(gid) if not exact else set()
        anc_match = bool(ancs & uniprot_canon)
        desc_match = bool(desc & uniprot_canon)
        consistent = exact or anc_match or desc_match

        rec = {
            "uniprot_id":      uid,
            "qa_hash":         c["qa_hash"],
            "aspect":          aspect,
            "claim":           c["claim"],
            "claim_go_id":     gid,
            "claim_go_name":   c["go_name"],
            "uniprot_n_terms": len(uniprot_canon),
            "exact_match":     exact,
            "ancestor_match":  anc_match,
            "descendant_match": desc_match,
            "consistent":      consistent,
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not consistent:
            miss_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_mismatch += 1

        s = counts[aspect]
        s["n"] += 1
        s["exact"] += int(exact)
        s["anc"]   += int(anc_match)
        s["desc"]  += int(desc_match)
        s["consistent"] += int(consistent)
        if not uniprot_canon:
            s["no_uniprot_terms"] += 1
    out_fh.close()
    miss_fh.close()

    # ---- summary ----
    print()
    print(f"=== GO consistency ({args.claims}) ===")
    for aspect in sorted(counts):
        s = counts[aspect]
        n = s["n"]
        print(f"\n[{aspect}] {ASPECT_NAME[aspect]} — {n} claims")
        print(f"  exact match:           {s['exact']:>6} ({100*s['exact']/n:.2f}%)")
        if not args.skip_ancestor:
            print(f"  ancestor match (only): {s['anc'] - s['exact']:>6}")
            print(f"  descendant match (only):{s['desc'] - s['exact']:>6}")
            print(f"  consistent (any):      {s['consistent']:>6} ({100*s['consistent']/n:.2f}%)")
        print(f"  proteins w/ 0 UniProt {aspect} terms: {s['no_uniprot_terms']}  "
              f"({100*s['no_uniprot_terms']/n:.2f}% of claims)")
    print(f"\n[done] per-claim verdicts: {args.output}")
    print(f"[done] mismatches ({n_mismatch}): {args.mismatch}")
    print(f"[done] uniprot cache:      {args.uniprot_cache}")


if __name__ == "__main__":
    main()
