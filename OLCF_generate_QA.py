"""Generate protein QA pairs from research abstracts using a local LLM.

For each entry in the input JSON (keyed by PubMed_ID + abstract), prompt the
model to extract every protein mentioned in the abstract and produce one
sequence-derivable (Question, Answer) conversation per protein.

Output is a JSON list, one entry per input article:
    [
      {"PubMed_ID": "...", "response": "<raw model output containing the JSON block>"}
    ]

Resumable — when --output exists, already-processed PubMed_IDs are skipped.

Usage:
    python OLCF_generate_QA.py --model <hf-model-id> --input <articles.json> \\
                               --output <out.json>
    python OLCF_generate_QA.py --model meta-llama/Llama-3.1-8B-Instruct \\
                               --input articles.json --output qa.json \\
                               --max-new-tokens 1500
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """We are building an AI predictor that predicts protein attributes from the sequence. We need your help to read a research abstract and generate questions about proteins that can be predicted from the sequence and that the abstract must have answers for.

Do not introduce yourself. Do not include greetings. Be concise, formal, and focused.

You will be given:
1. A research abstract.

Your task is to:
- For each protein name that appears in the abstract,
- Generate **one conversation** about that protein.
- If no relevant question-answer pairs can be created, return `conversations: null` for that protein.

Conversation rules:
- Only create question-and-answer pairs that can be answered using the protein sequence alone, ensuring each pair teaches the model how to infer that protein's attributes or functions from its sequence.
- Generate as many question-answer pairs as possible, but ensure they are relevant to the protein and the abstract.
- The conversation should be about the specific protein and must be fully based on information available in the abstract only.
- Do not invent information. If a question cannot be answered using the abstract, do not include it.
- Do not mention the word "abstract" or refer to the source of the information in any way.
- The answers must be factual, grounded only in the content of the abstract.
- Skip speculative, vague, or hypothetical answers.

Output format:
```json
[
  {
    "protein": "PROTEIN_NAME",
    "conversations": [
      {"Question": "What is its function?", "Answer": "It acts as a transcriptional co-activator involved in cell proliferation."},
      {"Question": "...", "Answer": "..."}
    ]
  },
  {
    "protein": "PROTEIN_NAME_2",
    "conversations": null
  }
]
```

Repeat the above process for every protein in the list.
Do not add any text or commentary outside of the specified format.
"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def initialize_model(model_id, device, dtype=torch.float16):
    """Load tokenizer + model and move the model to `device`."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True,
                                              trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, system_prompt, user_content):
    """Use the tokenizer's chat template when available (correct for Llama-3,
    Qwen, DeepSeek, etc.); fall back to a simple string otherwise."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"{system_prompt.strip()}\n\n{user_content.strip()}\n"


@torch.no_grad()
def run_inference(tokenizer, model, device, prompt, max_new_tokens=1500,
                  temperature=0.7, top_p=0.95, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Return only the model's continuation, not the echoed prompt.
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"input must be a JSON list, got {type(data).__name__}")
    return data


def load_done(path):
    """Read output file (if it exists) and return set of already-done PubMed_IDs."""
    if not Path(path).exists():
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            prior = json.load(f)
        if not isinstance(prior, list):
            return [], set()
        done = {str(r.get("PubMed_ID")) for r in prior if r.get("PubMed_ID") is not None}
        return prior, done
    except (json.JSONDecodeError, OSError):
        return [], set()


def safe_write_json(path, obj):
    """Atomic write via tmp file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def slugify_for_filename(s):
    """Make a model id safe to put in a filename: replace path separators."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True,
                   help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--input",  required=True,
                   help="input JSON: list of records with PubMed_ID + abstract")
    p.add_argument("--output", default=None,
                   help="output JSON path; defaults to ./generated_conversations_<model>.json")
    p.add_argument("--max-new-tokens", type=int, default=1500)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top-p",          type=float, default=0.95)
    p.add_argument("--top-k",          type=int,   default=50)
    p.add_argument("--save-every",     type=int,   default=10,
                   help="flush output to disk every N processed articles")
    p.add_argument("--device",         default=None,
                   help="cuda / cuda:0 / cpu (auto-detected if omitted)")
    args = p.parse_args()

    if args.output is None:
        args.output = f"./generated_conversations_{slugify_for_filename(args.model)}.json"

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[info] device: {device}", flush=True)
    print(f"[info] model:  {args.model}", flush=True)

    tokenizer, model = initialize_model(args.model, device)

    print(f"[info] loading: {args.input}", flush=True)
    articles = load_input(args.input)
    print(f"[info]   {len(articles)} articles", flush=True)

    output_responses, done = load_done(args.output)
    if done:
        print(f"[info] resume: {len(done)} PubMed_IDs already in {args.output}",
              flush=True)

    n_done_this_run = 0
    for entry in articles:
        pmid = str(entry.get("PubMed_ID", ""))
        abstract = entry.get("abstract", "")
        if not pmid or not abstract:
            continue
        if pmid in done:
            continue

        prompt = build_prompt(tokenizer, SYSTEM_PROMPT, abstract)
        try:
            response = run_inference(
                tokenizer, model, device, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        except Exception as e:
            print(f"  [warn] inference failed for PubMed_ID={pmid}: {e}",
                  file=sys.stderr)
            response = ""

        output_responses.append({"PubMed_ID": pmid, "response": response})
        done.add(pmid)
        n_done_this_run += 1

        if n_done_this_run % args.save_every == 0:
            safe_write_json(args.output, output_responses)
            print(f"  [progress] {len(output_responses)}/{len(articles)}  "
                  f"(+{n_done_this_run} this run)", flush=True)

    safe_write_json(args.output, output_responses)
    print(f"[done] {len(output_responses)} records written to {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
