#!/usr/bin/env python3
"""Aggregate multiple cleaner-LLM outputs into a unanimous-consensus QA set
and filter the original QA JSONL accordingly.

Each cleaner output is a JSON array of records of the form
    {"PubMed_ID": ..., "proteins": [{"protein": ..., "conversations": [
        {"Question": ..., "QC": "pass" | "fail"}, ...
    ]}]}

A (PubMed_ID, protein, Question) triple is kept only if EVERY cleaner marks
its QC as 'pass'. Otherwise it's flagged as a hallucination candidate.

Usage:
    python clean_and_merge_2.py \\
        --cleaner-outputs cleaner1.json cleaner2.json cleaner3.json \\
        --original-jsonl  path/to/original_qa.jsonl \\
        --out-dir         consensus_results
"""

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

# ---------------------------
# Helpers
# ---------------------------

def _norm(s: str) -> str:
    """Normalize for comparison: casefold, collapse whitespace, trim."""
    if s is None:
        return ""
    s = s.casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _short(s: str, n: int = 80) -> str:
    """Shorten long strings for logging."""
    if s is None:
        return "None"
    s = s.strip()
    return s if len(s) <= n else s[:n-3] + "..."

def _coerce_conv_item(item):
    """
    Ensure a conversation item is a dict. If it's a string that looks like JSON, parse it.
    Return dict or None if unusable.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        t = item.strip()
        if t.startswith("{") and t.endswith("}"):
            try:
                obj = json.loads(t)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None

# ---------------------------
# Consensus filter (with tqdm)
# ---------------------------

def filter_all_pass_else_fail_qas_across_models_v2(file_paths):
    """
    For each question (PubMedID, protein, Question) present in all files:
      - QC = 'pass' only if ALL files mark it 'pass'; otherwise 'fail'.

    Prints warnings when proteins/questions are dropped due to exact-match sensitivity
    but would have matched after normalization (case/space differences).

    Returns:
        - result_qas: List[dict] with {"PubMed_ID","protein","Question","QC"}
        - hallucination_rate: (# fail) / (total questions)
        - hallucinated_examples: subset of result_qas with QC == 'fail'
    """
    all_data = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            all_data.append(json.load(f))

    # PubMed_ID assumed unique and always present
    pubmed_sets = [set(entry["PubMed_ID"] for entry in data) for data in all_data]
    common_pubmed_ids = set.intersection(*pubmed_sets)

    result_qas = []
    hallucinated_examples = []
    n_fail = 0

    for pmid in tqdm(sorted(common_pubmed_ids), desc="Aggregating consensus across PMIDs", unit="pmid"):
        # gather per-file entry for this pmid
        entries = []
        for data in all_data:
            entry = next((e for e in data if e.get("PubMed_ID") == pmid), None)
            if entry is None:
                entries = []
                break
            entries.append(entry)
        if not entries:
            continue

        # ---- Proteins: exact set vs normalized set (for diagnostics) ----
        protein_sets_exact = [set(
            p.get("protein") for p in e.get("proteins", []) if isinstance(p, dict) and p.get("protein") is not None
        ) for e in entries]
        common_proteins_exact = set.intersection(*protein_sets_exact) if protein_sets_exact else set()

        # Build normalized maps to catch near-equal items
        protein_maps_norm = []
        for e in entries:
            m = {}
            for p in e.get("proteins", []):
                if not isinstance(p, dict):
                    continue
                name = p.get("protein")
                if not isinstance(name, str):
                    continue
                key = _norm(name)
                m.setdefault(key, set()).add(name)
            protein_maps_norm.append(m)

        common_proteins_norm = set.intersection(*(set(m.keys()) for m in protein_maps_norm)) if protein_maps_norm else set()

        # Warn when normalization reveals agreement but exact strings differ
        for nk in sorted(common_proteins_norm):
            originals_per_file = [protein_maps_norm[i].get(nk, set()) for i in range(len(entries))]
            candidates = set.intersection(*originals_per_file)
            if not candidates and not any(orig in common_proteins_exact for origs in originals_per_file for orig in origs):
                details = " | ".join(
                    f"f{i+1}:{', '.join(sorted(map(_short, s)) or ['—'])}"
                    for i, s in enumerate(originals_per_file)
                )
                print(f"[WARN:PROTEIN_EXACT_MATCH] PubMed_ID={pmid} "
                      f"Normalized protein '{_short(nk)}' differs across files. "
                      f"Exact match required; skipping. Variants -> {details}")

        # Proceed only with exact matches (original logic)
        for protein_name in common_proteins_exact:
            # collect the protein blocks for this exact protein_name from each entry
            protein_entries = []
            for entry in entries:
                protein = next((p for p in entry.get("proteins", []) if isinstance(p, dict) and p.get("protein") == protein_name), None)
                if protein is None:
                    protein_entries = []
                    break
                protein_entries.append(protein)
            if not protein_entries:
                continue

            # ---- Questions: exact vs normalized (diagnostics) ----
            # Build exact question sets safely
            question_sets_exact = []
            for p in protein_entries:
                qs = set()
                for c in p.get("conversations", []):
                    c_obj = _coerce_conv_item(c)
                    if c_obj is None:
                        continue
                    q = c_obj.get("Question")
                    if isinstance(q, str):
                        qs.add(q)
                question_sets_exact.append(qs)

            common_questions_exact = set.intersection(*question_sets_exact) if question_sets_exact else set()

            # Normalized maps for questions
            question_maps_norm = []
            for p in protein_entries:
                qmap = {}
                for c in p.get("conversations", []):
                    c_obj = _coerce_conv_item(c)
                    if c_obj is None:
                        continue
                    q = c_obj.get("Question")
                    if not isinstance(q, str):
                        continue
                    qk = _norm(q)
                    qmap.setdefault(qk, set()).add(q)
                question_maps_norm.append(qmap)

            common_questions_norm = set.intersection(*(set(m.keys()) for m in question_maps_norm)) if question_maps_norm else set()

            # Warn if normalization reveals matches that exact comparison drops
            for qk in sorted(common_questions_norm):
                originals_per_file = [question_maps_norm[i].get(qk, set()) for i in range(len(protein_entries))]
                candidates = set.intersection(*originals_per_file)
                if not candidates and not any(orig in common_questions_exact for origs in originals_per_file for orig in origs):
                    details = " | ".join(
                        f"f{i+1}:{' || '.join(sorted(map(lambda x: _short(x, 120), s)) or ['—'])}"
                        for i, s in enumerate(originals_per_file)
                    )
                    print(f"[WARN:QUESTION_EXACT_MATCH] PubMed_ID={pmid} protein='{_short(protein_name)}' "
                          f"Normalized question differs across files. "
                          f"Exact match required; skipping. Variants -> {details}")

            # ---- Evaluate exact-common questions (unchanged logic) ----
            for question in common_questions_exact:
                qcs = []
                skip_question = False
                for protein in protein_entries:
                    # find matching conv safely
                    conv = None
                    for c in protein.get("conversations", []):
                        c_obj = _coerce_conv_item(c)
                        if c_obj is None:
                            continue
                        if c_obj.get("Question") == question:
                            conv = c_obj
                            break
                    if conv is None:
                        skip_question = True
                        break
                    qc_str = str(conv.get("QC", "")).strip().lower()
                    qcs.append(qc_str)
                if skip_question:
                    continue

                qc_final = "pass" if all(q == "pass" for q in qcs) else "fail"

                qa_result = {
                    "PubMed_ID": pmid,
                    "protein": protein_name,
                    "Question": question,
                    "QC": qc_final
                }
                result_qas.append(qa_result)

                if qc_final == "fail":
                    hallucinated_examples.append(qa_result)
                    n_fail += 1

    total = len(result_qas)
    hallucination_rate = n_fail / total if total > 0 else 0.0
    return result_qas, hallucination_rate, hallucinated_examples

# ---------------------------
# Filter original JSONL (with tqdm)
# ---------------------------

def keep_consensus_pass_in_original(
    original_jsonl_path: str,
    result_qas: list,
    output_jsonl_path: str
):
    """
    Filter the original JSONL to only those (PubMed_ID, protein, Question) unanimously 'pass'.
    Saves to output_jsonl_path.
    """
    passed = {
        (r["PubMed_ID"], r["protein"], r["Question"])
        for r in result_qas
        if str(r.get("QC", "")).strip().lower() == "pass"
    }

    in_path = Path(original_jsonl_path)
    out_path = Path(output_jsonl_path)

    kept_records = 0
    kept_convs = 0

    # Count for progress bar
    with open(in_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout, \
         tqdm(total=total_lines, desc="Filtering original JSONL", unit="line") as pbar:
        for line_num, line in enumerate(fin, start=1):
            pbar.update(1)
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pmid = rec.get("PubMed_ID")
                proteins = rec.get("proteins", [])

                new_proteins = []
                for p in proteins:
                    if not isinstance(p, dict):
                        continue
                    pname = p.get("protein")
                    convs = p.get("conversations", [])
                    new_convs = []
                    for c in convs:
                        c_obj = c if isinstance(c, dict) else _coerce_conv_item(c)
                        if c_obj is None:
                            continue
                        if (pmid, pname, c_obj.get("Question")) in passed:
                            new_convs.append(c_obj)
                    if new_convs:
                        kept_convs += len(new_convs)
                        new_p = dict(p)
                        new_p["conversations"] = new_convs
                        new_proteins.append(new_p)

                if new_proteins:
                    kept_records += 1
                    out_obj = dict(rec)
                    out_obj["proteins"] = new_proteins
                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"[WARN] Skipping line {line_num} due to error: {e}")

    print(f"[DONE] Wrote: {out_path}")
    print(f"       Records kept: {kept_records}")
    print(f"       Conversations kept: {kept_convs}")

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build unanimous-consensus QA set from N cleaner outputs and "
                    "filter the original QA JSONL accordingly."
    )
    parser.add_argument(
        "--cleaner-outputs", nargs="+", required=True,
        help="paths to cleaner LLM output files (JSON arrays); at least 2",
    )
    parser.add_argument(
        "--original-jsonl", required=True,
        help="path to the original (pre-QC) QA JSONL to filter",
    )
    parser.add_argument(
        "--out-dir", default="consensus_results",
        help="directory to write consensus artifacts (default: consensus_results)",
    )
    parser.add_argument(
        "--filtered-name", default="consensus_passed.jsonl",
        help="name of the filtered output JSONL inside --out-dir",
    )
    args = parser.parse_args()

    if len(args.cleaner_outputs) < 2:
        parser.error("--cleaner-outputs needs at least 2 files for a consensus")

    # Compute consensus across the supplied cleaner outputs
    result_qas, hallucination_rate, hallucinated_examples = \
        filter_all_pass_else_fail_qas_across_models_v2(args.cleaner_outputs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "consensus_qas.json", "w", encoding="utf-8") as f:
        json.dump(result_qas, f, ensure_ascii=False, indent=2)

    with open(out_dir / "hallucinated_examples.json", "w", encoding="utf-8") as f:
        json.dump(hallucinated_examples, f, ensure_ascii=False, indent=2)

    with open(out_dir / "hallucination_rate.txt", "w", encoding="utf-8") as f:
        f.write(f"Hallucination rate: {hallucination_rate:.4f}\n")
        f.write(f"Total QAs: {len(result_qas)}\n")
        f.write(f"Hallucinated: {len(hallucinated_examples)}\n")
        f.write(f"Cleaners used: {len(args.cleaner_outputs)}\n")

    print(f"[DONE] Consensus results saved to: {out_dir}")
    print(f"       Total QAs: {len(result_qas)}")
    print(f"       Hallucination rate: {hallucination_rate:.4f}")

    # Filter the original JSONL to only consensus-passed QAs
    keep_consensus_pass_in_original(
        original_jsonl_path=args.original_jsonl,
        result_qas=result_qas,
        output_jsonl_path=out_dir / args.filtered_name,
    )
