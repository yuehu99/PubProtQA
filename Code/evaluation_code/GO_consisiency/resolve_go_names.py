"""Stage 1.5: resolve LLM-proposed GO term names to real GO ids using the
official ontology (go-basic.obo).

We do not trust GO ids the LLM emits, but the term name is usually close.
Lookup cascade for every BP/MF/CC claim with a non-null go_name:
    1. exact normalized match against primary term names
    2. exact normalized match against EXACT / NARROW synonyms
    3. *generalize* the name (strip qualifiers, prepositional tails,
    leading "(positive|negative) regulation of") and retry 1+2 on each
    progressively more general variant — this lets a too-specific LLM
    phrase like "retrograde endosome transport into the hyphal apex"
    fall back to a real GO term such as "endosome transport".
    4. (optional) fuzzy match via difflib above a threshold

Output is a parallel JSONL where each claim gains:
    "go_id":            "GO:0006915" | null
    "go_name_resolved": canonical name found by the lookup | null
    "lookup_method":    "exact_name" | "exact_synonym" | "narrow_synonym"
                        | "generalized" | "fuzzy" | "not_found" | "no_name"
    "lookup_query":     the variant that produced the hit | null
    "lookup_score":     float | null   (fuzzy only)

Usage:
    python resolve_go_names.py --claims <claims.jsonl> --output <out.jsonl> --obo <go-basic.obo>
    python resolve_go_names.py --claims ... --output ... --obo ... --fuzzy --fuzzy-threshold 0.88
    python resolve_go_names.py --claims ... --output ... --obo ... --no-generalize
"""

import argparse
import difflib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen


GO_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"

_GO_TYPES = {"biological_process", "molecular_function", "cellular_component"}
TYPE_TO_ASPECT = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}

_WS = re.compile(r"\s+")
_NON_WORD_EDGE = re.compile(r"^[^\w]+|[^\w]+$")


def normalize(s):
    if not s:
        return ""
    s = s.strip().lower()
    s = _WS.sub(" ", s)
    return s


_PREP_LAST = re.compile(r"\s+(?:into|onto|to|from|via|by|in)\s+", re.I)
_PARENS_TAIL = re.compile(r"\s*\([^)]*\)\s*$")
_INVOLVED_IN = re.compile(r"\s+involved\s+in\s+\S+.*$", re.I)
_DURING_TAIL = re.compile(r"\s+during\s+\S+.*$", re.I)
_NEG_POS_REG = re.compile(r"^(?:positive|negative)\s+regulation\s+of\s+", re.I)
_REG_OF      = re.compile(r"^regulation\s+of\s+", re.I)


def _generalize_step(cur):
    """Apply ONE simplification step; return the new string or None."""
    # 1. Trailing parenthetical
    m = _PARENS_TAIL.search(cur)
    if m and m.start() > 0:
        return cur[:m.start()].rstrip()
    # 2. "...involved in X"
    m = _INVOLVED_IN.search(cur)
    if m:
        return cur[:m.start()].rstrip()
    # 3. "...during X"
    m = _DURING_TAIL.search(cur)
    if m:
        return cur[:m.start()].rstrip()
    # 4. Strip the RIGHTMOST comma-tail: "X, foo"
    idx = cur.rfind(", ")
    if idx > 0:
        return cur[:idx].rstrip()
    # 5. Strip the RIGHTMOST trailing prepositional phrase
    last = None
    for m in _PREP_LAST.finditer(cur):
        last = m
    if last:
        return cur[:last.start()].rstrip()
    # 6. "(positive|negative) regulation of X" -> "regulation of X"
    m = _NEG_POS_REG.match(cur)
    if m:
        return "regulation of " + cur[m.end():]
    # 7. "regulation of X" -> "X"
    m = _REG_OF.match(cur)
    if m:
        return cur[m.end():]
    return None


def generalize(name):
    """Yield (variant, label) progressively simpler variants of `name`.

    The first variant is the original; each subsequent variant applies one
    simplification step. Stops when no rule applies or we'd produce a
    duplicate / empty string.
    """
    if not name:
        return
    cur = _WS.sub(" ", name.strip())
    if not cur:
        return
    seen = {cur.lower()}
    yield cur, "original"
    while True:
        nxt = _generalize_step(cur)
        if nxt is None:
            return
        nxt = _WS.sub(" ", nxt).strip()
        if not nxt or nxt.lower() in seen:
            return
        seen.add(nxt.lower())
        yield nxt, "generalized"
        cur = nxt


def ensure_obo(path):
    if Path(path).exists():
        return
    print(f"[info] downloading go-basic.obo -> {path}")
    with urlopen(Request(GO_OBO_URL), timeout=120) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)


_SYN_RE = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"\s+(\S+)')


def parse_obo_names(path):
    """Return (name_index, term_info) where:

    name_index: {normalized_name: [ (go_id, source, aspect, canonical_name) ... ]}
        source ∈ {"name", "exact_synonym", "narrow_synonym"}
    term_info:  {go_id: {"name": canonical_name, "aspect": "P"|"F"|"C"}}
    """
    name_index = defaultdict(list)
    term_info = {}
    cur = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("[Term]"):
                cur = {"id": None, "name": None, "ns": None, "obs": False, "syns": []}
            elif line.startswith("["):
                # leaving Term stanza
                _flush(cur, name_index, term_info)
                cur = None
            elif cur is not None:
                if line.startswith("id: "):
                    cur["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    cur["name"] = line[6:].strip()
                elif line.startswith("namespace: "):
                    ns = line[11:].strip()
                    cur["ns"] = {"biological_process": "P",
                                "molecular_function": "F",
                                "cellular_component": "C"}.get(ns)
                elif line.startswith("is_obsolete: true"):
                    cur["obs"] = True
                elif line.startswith("synonym:"):
                    m = _SYN_RE.match(line)
                    if m:
                        text, scope = m.group(1), m.group(2)
                        if scope in ("EXACT", "NARROW"):
                            cur["syns"].append((scope.lower(), text))
                elif line == "":
                    _flush(cur, name_index, term_info)
                    cur = None
        _flush(cur, name_index, term_info)
    return name_index, term_info


def _flush(cur, name_index, term_info):
    if not cur or cur.get("obs"):
        return
    if not (cur.get("id") and cur.get("name") and cur.get("ns")):
        return
    gid = cur["id"]
    name = cur["name"]
    aspect = cur["ns"]
    term_info[gid] = {"name": name, "aspect": aspect}
    name_index[normalize(name)].append((gid, "name", aspect, name))
    for scope, text in cur["syns"]:
        src = "exact_synonym" if scope == "exact" else "narrow_synonym"
        name_index[normalize(text)].append((gid, src, aspect, name))


def best_exact(name_index, query, want_aspect):
    """Return (go_id, source, canonical_name) or None."""
    hits = name_index.get(query)
    if not hits:
        return None
    # Prefer matches in the desired aspect, then primary names over synonyms.
    src_priority = {"name": 0, "exact_synonym": 1, "narrow_synonym": 2}
    hits = sorted(hits, key=lambda h: (h[2] != want_aspect, src_priority[h[1]]))
    gid, src, _aspect, canon = hits[0]
    return gid, src, canon


def fuzzy_lookup(all_keys_for_aspect, query, threshold):
    """Return (best_key, score) or (None, None)."""
    matches = difflib.get_close_matches(query, all_keys_for_aspect, n=1, cutoff=threshold)
    if not matches:
        return None, None
    best = matches[0]
    score = difflib.SequenceMatcher(None, query, best).ratio()
    return best, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True,
                    help="extracted-claims jsonl from extract_go_claims.py")
    ap.add_argument("--output", required=True,
                    help="path to write resolved-claims jsonl")
    ap.add_argument("--obo",    required=True,
                    help="path to go-basic.obo (downloaded if missing)")
    ap.add_argument("--fuzzy",  action="store_true",
                    help="fall back to difflib fuzzy match if exact / synonym miss")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.88,
                    help="fuzzy match cutoff in [0,1] (default 0.88)")
    ap.add_argument("--no-generalize", action="store_true",
                    help="disable text-level name generalization fallback")
    args = ap.parse_args()

    ensure_obo(args.obo)
    print(f"[info] parsing OBO: {args.obo}")
    t0 = time.time()
    name_index, term_info = parse_obo_names(args.obo)
    print(f"[info] {len(term_info)} non-obsolete terms, "
        f"{len(name_index)} unique normalized name keys "
        f"(parsed in {time.time() - t0:.1f}s)")

    # Aspect-partitioned key lists for fuzzy fallback.
    keys_by_aspect = {"P": [], "F": [], "C": []}
    if args.fuzzy:
        for key, hits in name_index.items():
            seen = set()
            for _gid, _src, aspect, _canon in hits:
                if aspect in keys_by_aspect and aspect not in seen:
                    keys_by_aspect[aspect].append(key)
                    seen.add(aspect)
        print(f"[info] fuzzy enabled (threshold={args.fuzzy_threshold}), "
            f"keys: P={len(keys_by_aspect['P'])} "
            f"F={len(keys_by_aspect['F'])} "
            f"C={len(keys_by_aspect['C'])}")

    # ---- pass over claims jsonl ----
    n_in = 0
    n_go_claim = 0
    method_counts = defaultdict(int)

    with open(args.claims, "r", encoding="utf-8") as fin, \
        open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            new_claims = []
            for c in rec.get("claims", []):
                ctype = c.get("type")
                if ctype not in _GO_TYPES:
                    new_claims.append(c)
                    continue
                n_go_claim += 1
                aspect = TYPE_TO_ASPECT[ctype]
                go_name_in = c.get("go_name")
                resolved = None
                method = None
                score = None

                used_query = None
                if not go_name_in:
                    method = "no_name"
                else:
                    # Build the cascade: original first, then progressively
                    # more general variants (unless disabled).
                    if args.no_generalize:
                        variants = [(go_name_in, "original")]
                    else:
                        variants = list(generalize(go_name_in))
                    for i, (variant, _label) in enumerate(variants):
                        q = normalize(variant)
                        if not q:
                            continue
                        hit = best_exact(name_index, q, aspect)
                        if hit:
                            gid, src, canon = hit
                            resolved = (gid, canon)
                            used_query = variant
                            if i == 0:
                                # First (original) hit — report its source.
                                method = "exact_name" if src == "name" else src
                            else:
                                # Hit only after simplification.
                                method = "generalized"
                            break
                    if not resolved and args.fuzzy and keys_by_aspect.get(aspect):
                        q = normalize(go_name_in)
                        best_key, score = fuzzy_lookup(
                            keys_by_aspect[aspect], q, args.fuzzy_threshold)
                        if best_key:
                            hit2 = best_exact(name_index, best_key, aspect)
                            if hit2:
                                gid, _src, canon = hit2
                                resolved = (gid, canon)
                                method = "fuzzy"
                                used_query = best_key
                    if not resolved and method is None:
                        method = "not_found"

                method_counts[method] += 1
                new_c = dict(c)
                new_c["go_id"] = resolved[0] if resolved else None
                new_c["go_name_resolved"] = resolved[1] if resolved else None
                new_c["lookup_method"] = method
                new_c["lookup_query"] = used_query
                new_c["lookup_score"] = score
                new_claims.append(new_c)

            rec["claims"] = new_claims
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- report ----
    print()
    print(f"=== resolved {n_in} records, {n_go_claim} GO-typed claims ===")
    width = max((len(k) for k in method_counts), default=10)
    for m in ("exact_name", "exact_synonym", "narrow_synonym", "generalized",
            "fuzzy", "not_found", "no_name"):
        n = method_counts.get(m, 0)
        pct = 100 * n / n_go_claim if n_go_claim else 0
        print(f"  {m:<{width}}  {n:>7}  ({pct:5.2f}%)")
    print(f"\n[done] resolved claims -> {args.output}")


if __name__ == "__main__":
    main()
