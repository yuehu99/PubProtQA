#!/usr/bin/env python3
"""LLM-based verification pipeline for grouped-by-protein JSON datasets.

Uses OpenRouter (Gemini 2.5 Flash by default) with multiprocessing
parallelism and shard/resume support.

Usage:
    python run_llm_pipeline.py --input_file /path/to/dataset.json --sample 100 --merge_output
    python run_llm_pipeline.py --input_file /path/to/dataset.json --n_process 5 --merge_output
    python run_llm_pipeline.py --input_file /path/to/dataset.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Process, current_process
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUTS_DIR, PROJECT_ROOT
from fact_builder import build_facts
from llm_verifier import (
    DEFAULT_MODEL,
    aggregate_label,
    verify_qa_llm,
)
import uniprot_fetcher

DEFAULT_API_KEYS_FILE = PROJECT_ROOT / "api_keys.txt"


def read_api_keys(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def safe_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "model"


def flatten_proteins(data: dict) -> list[tuple[int, str, dict]]:
    """Flatten protein dict into a list of (global_qa_index, uid, qa_dict) tuples."""
    items = []
    idx = 0
    for uid, prot in data.items():
        seq = prot.get("seq", "")
        convos = prot.get("conversations", [])
        if not seq or not convos:
            continue
        for qa in convos:
            q = qa.get("question", "") or qa.get("Question", "")
            a = qa.get("answer", "") or qa.get("Answer", "")
            if not q or not a:
                continue
            items.append((idx, uid, {
                "question": q,
                "answer": a,
                "seq": seq,
                "protein_names": prot.get("protein_names", [uid]),
            }))
            idx += 1
    return items


def process_chunk(items: list[tuple[int, str, dict]], model: str,
                  shard_path: str, api_key: str, batch_size: int,
                  dry_run: bool):
    """Worker: process a chunk of QA pairs, writing JSONL to shard file."""
    processed = set()
    if os.path.exists(shard_path):
        with open(shard_path) as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if "index" in j:
                        processed.add(j["index"])
                except Exception:
                    continue

    os.makedirs(os.path.dirname(os.path.abspath(shard_path)), exist_ok=True)

    uid_to_items: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for gidx, uid, qa_data in items:
        uid_to_items[uid].append((gidx, qa_data))

    facts_cache: dict[str, dict] = {}

    with open(shard_path, "a") as out_f:
        pbar = tqdm(items, desc=f"{current_process().name}", position=0)
        for gidx, uid, qa_data in pbar:
            if gidx in processed:
                pbar.update(0)
                continue

            if uid not in facts_cache:
                seq = qa_data["seq"]
                try:
                    uniprot_fetcher.lookup_batch([uid])
                    facts_cache[uid] = build_facts(seq, uid)
                except Exception:
                    facts_cache[uid] = {
                        "accession": uid,
                        "protein_length": len(seq),
                        "molecular_weight_da": None, "molecular_weight_kda": None,
                        "amino_acid_composition": {}, "organism": None,
                        "subcellular_locations": [], "function_text": None,
                        "ec_numbers": [], "domains": [], "active_sites": [],
                        "ptm_sites": [], "signal_peptide": None,
                        "transit_peptide": None, "subunit_text": None,
                        "keywords": [], "interpro_domains": [],
                        "transmembrane": [], "disulfide_bonds": [],
                        "secondary_structure": [], "sequence_length_uniprot": None,
                        "mol_weight_uniprot": None, "mature_protein_length": len(seq),
                    }

            facts = facts_cache[uid]
            q = qa_data["question"]
            a = qa_data["answer"]
            protein_name = qa_data["protein_names"][0] if qa_data["protein_names"] else uid

            if dry_run:
                record = {
                    "index": gidx,
                    "uniprot_id": uid,
                    "protein": protein_name,
                    "question": q,
                    "answer": a[:500],
                    "dry_run": True,
                }
            else:
                try:
                    verdicts = verify_qa_llm(q, a, facts, api_key=api_key, model=model)
                    agg = aggregate_label(verdicts) if verdicts else "not_applicable"

                    record = {
                        "index": gidx,
                        "uniprot_id": uid,
                        "protein": protein_name,
                        "question": q,
                        "answer": a[:500],
                        "aggregate": agg,
                        "verdicts": verdicts,
                        "n_claims": len(verdicts),
                    }
                except Exception as e:
                    record = {
                        "index": gidx,
                        "uniprot_id": uid,
                        "protein": protein_name,
                        "question": q,
                        "error": str(e),
                    }

            out_f.write(json.dumps(record, default=str) + "\n")
            out_f.flush()


def merge_shards(shard_paths: list[str], merged_path: str) -> list[dict]:
    rows = []
    for p in shard_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    rows.sort(key=lambda x: x.get("index", 10**12))
    with open(merged_path, "w") as out_f:
        for r in rows:
            out_f.write(json.dumps(r, default=str) + "\n")
    return rows


def compute_stats(rows: list[dict]) -> dict:
    category_counts: dict[str, Counter] = defaultdict(Counter)
    aggregate_counts = Counter()
    total_qa = 0
    total_claims = 0
    errors = 0

    for r in rows:
        if "error" in r:
            errors += 1
            continue
        if "aggregate" not in r:
            continue
        total_qa += 1
        aggregate_counts[r["aggregate"]] += 1
        for v in r.get("verdicts", []):
            cat = v.get("question_category", "other")
            category_counts[cat][v["status"]] += 1
            total_claims += 1

    return {
        "total_qa": total_qa,
        "total_claims": total_claims,
        "errors": errors,
        "aggregate_counts": dict(aggregate_counts),
        "category_counts": {cat: dict(counts) for cat, counts in category_counts.items()},
    }


def write_summary(stats: dict, path: str, split_name: str):
    category_counts = stats["category_counts"]
    aggregate_counts = stats["aggregate_counts"]
    total_qa = stats["total_qa"]
    total_claims = stats["total_claims"]

    lines = [f"# {split_name.capitalize()} — LLM Verification Summary\n"]
    lines.append(f"- QA pairs processed: {total_qa:,}")
    lines.append(f"- Total claims extracted: {total_claims:,}")
    lines.append(f"- Errors: {stats['errors']:,}")
    lines.append("")

    lines.append("## Aggregate Verdicts\n")
    lines.append("| Verdict | Count | % |")
    lines.append("|---------|-------|---|")
    for verdict in ["verified", "contradicted", "not_applicable"]:
        c = aggregate_counts.get(verdict, 0)
        pct = c / max(total_qa, 1) * 100
        lines.append(f"| {verdict} | {c:,} | {pct:.1f}% |")
    lines.append("")

    lines.append("## Per-Category Breakdown\n")
    lines.append("| Category | Verified | Contradicted | N/A | Total | Precision |")
    lines.append("|----------|----------|-------------|-----|-------|-----------|")

    all_v = all_c = all_na = 0
    for cat in sorted(category_counts.keys()):
        counts = category_counts[cat]
        v = counts.get("verified", 0)
        c = counts.get("contradicted", 0)
        na = counts.get("not_applicable", 0)
        total = v + c + na
        prec = v / (v + c) if (v + c) > 0 else 0.0
        lines.append(f"| {cat} | {v:,} | {c:,} | {na:,} | {total:,} | {prec:.1%} |")
        all_v += v
        all_c += c
        all_na += na

    total = all_v + all_c + all_na
    prec = all_v / (all_v + all_c) if (all_v + all_c) > 0 else 0.0
    lines.append(f"| **TOTAL** | **{all_v:,}** | **{all_c:,}** | **{all_na:,}** | **{total:,}** | **{prec:.1%}** |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def process_dataset(input_path: Path, args):
    split_name = input_path.stem

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = OUTPUTS_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data: dict[str, dict] = json.load(f)

    print(f"[{split_name}] {len(data)} proteins loaded from {input_path}")

    items = flatten_proteins(data)
    print(f"[{split_name}] {len(items)} QA pairs extracted")

    if args.sample > 0 and args.sample < len(items):
        import random
        random.seed(42)
        indices = sorted(random.sample(range(len(items)), args.sample))
        items = [items[i] for i in indices]
        print(f"[{split_name}] Sampled {len(items)} QA pairs")

    api_keys = read_api_keys(args.api_keys_file)
    n_keys = len(api_keys)
    total_workers = n_keys * args.n_process
    print(f"[{split_name}] {n_keys} API key(s), {args.n_process} process(es)/key "
          f"= {total_workers} worker(s)")

    model_tag = safe_tag(args.model)
    base = str(out_dir / "verification_results")
    shard_paths = []
    processes = []

    start_time = time.time()

    for ki in range(n_keys):
        for pi in range(args.n_process):
            wid = ki * args.n_process + pi
            chunk = items[wid::total_workers]
            if not chunk:
                continue
            sp = f"{base}_shard_{model_tag}_{wid}.jsonl"
            shard_paths.append(sp)
            p = Process(
                target=process_chunk,
                args=(chunk, args.model, sp, api_keys[ki],
                      args.batch_size, args.dry_run),
                name=f"w{wid}",
            )
            processes.append(p)

    print(f"[{split_name}] Starting {len(processes)} worker(s) ...")
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"[{split_name}] Workers done in {elapsed:.0f}s")

    if args.merge_output:
        merged_path = str(out_dir / "verification_results.jsonl")
        rows = merge_shards(shard_paths, merged_path)
        print(f"[{split_name}] Merged {len(rows)} results -> {merged_path}")

        stats = compute_stats(rows)
        summary_path = str(out_dir / "summary.md")
        write_summary(stats, summary_path, split_name)
        print(f"[{split_name}] Summary -> {summary_path}")

        stats_path = str(out_dir / "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        agg = stats["aggregate_counts"]
        tq = stats["total_qa"]
        print(f"\n[{split_name}] Results:")
        print(f"  QA pairs: {tq:,}  |  Claims: {stats['total_claims']:,}  |  Errors: {stats['errors']:,}")
        for verdict in ["verified", "contradicted", "not_applicable"]:
            c = agg.get(verdict, 0)
            pct = c / max(tq, 1) * 100
            print(f"  {verdict}: {c:,} ({pct:.1f}%)")
    else:
        print(f"[{split_name}] Done. Shard files: {shard_paths}")
        print("Run again with --merge_output to merge and compute stats.")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based verification on grouped-by-protein JSON datasets"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to grouped-by-protein JSON dataset")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Custom output directory (default: outputs/<dataset_name>/)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    parser.add_argument("--api_keys_file", type=str,
                        default=str(DEFAULT_API_KEYS_FILE),
                        help="Path to file with OpenRouter API keys (one per line)")
    parser.add_argument("--n_process", type=int, default=5,
                        help="Processes per API key (default: 5)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="UniProt batch lookup size")
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, randomly sample this many QA pairs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts but skip LLM calls")
    parser.add_argument("--merge_output", action="store_true",
                        help="Merge shard files and compute stats")
    args = parser.parse_args()

    path = Path(args.input_file)
    if not path.exists():
        print(f"Input file not found: {path}")
        sys.exit(1)

    process_dataset(path, args)


if __name__ == "__main__":
    main()
