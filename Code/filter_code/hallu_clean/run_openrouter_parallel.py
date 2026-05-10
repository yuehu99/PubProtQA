import os
import numpy
import requests
import json
import argparse
import re
from tqdm import tqdm  # Progress bar for serial processing
from multiprocessing import Process, current_process
import math

# ---------------------------------------------
# Helper functions for loading JSON and JSONL
# ---------------------------------------------

def load_from_jsonl(path):
    """Load a .jsonl file and return a list of parsed records."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def load_from_json(path):
    """Load a .json file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------------------------
# QA pair filtering & abstract attachment helpers (serial version)
# ---------------------------------------------

def prepare_qa_pairs(qa_pairs, articles):
    """Filter QA pairs, attach abstracts, and return the cleaned list."""
    article_lookup = {
        art["PubMed_ID"]: art.get("abstract", "").replace("\n", " ").strip()
        for art in articles
    }
    cleaned = []
    dropped = 0
    for qa_entry in qa_pairs:
        if not qa_entry.get("proteins") or not any(
            p.get("conversations") for p in qa_entry.get("proteins", [])
        ):
            continue
        pmid = qa_entry["PubMed_ID"]
        abstract = article_lookup.get(pmid)
        if abstract is None:
            dropped += 1
            continue
        qa_entry["abstract"] = abstract
        cleaned.append(qa_entry)
    print(f"Removed {dropped} entries, {len(cleaned)} entries remaining")
    return cleaned

# ---------------------------------------------
# Prompt & checkpoint utilities
# ---------------------------------------------

def build_prompt(abstract: str, proteins):
    """Construct the prompt string given an *abstract* and *proteins* list."""
    return f"Here is the abstract:\n{abstract}\n\nHere is the QA pairs:\n{proteins}"

def read_api_keys(api_keys_path):
    """Read API keys from a file, one per line, ignoring empty lines and whitespace."""
    with open(api_keys_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# ---------------------------------------------
# OpenRouter wrapper
# ---------------------------------------------

class OpenRouter:
    """
    This class is used to interact with the OpenRouter API.
    NOTE: specify API key as an os environment variable. 
    """
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_APIKEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_APIKEY environment variable not set")
        # Print masked confirmation only — never echo any portion of the key.
        print(f"API key loaded ({len(self.api_key)} chars).")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    def get_response(self, prompt: str, model: str = "moonshotai/kimi-k2:free"):
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        }
        response = requests.post(url=url,
                                headers=self.headers, 
                                data=json.dumps(payload))
        return response.json()

def process_chunk(chunk, model_name, role_prompt, output_file, api_key):
    """Process a chunk of QA pairs using a specific API key and write results to output_file."""
    os.environ["OPENROUTER_APIKEY"] = api_key
    router = OpenRouter()
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f_res:
            for line in f_res:
                try:
                    processed_ids.add(json.loads(line)["PubMed_ID"])
                except Exception:
                    continue
    with open(output_file, "a") as out_f:
        for qa_entry in tqdm(chunk, desc=f"{current_process().name} {model_name}"):
            pmid = qa_entry["PubMed_ID"]
            if pmid in processed_ids:
                continue
            prompt = build_prompt(qa_entry["abstract"], qa_entry.get("proteins", []))
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": role_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                response = requests.post(
                    url=url, headers=router.headers, data=json.dumps(payload)
                )
                response_json = response.json()
                if "error" in response_json:
                    raise RuntimeError(response_json["error"])
                content = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "No content")
                )
                record = {
                    "PubMed_ID": pmid,
                    "content": content,
                    "full_response": response_json,
                }
            except Exception as e:
                record = {"PubMed_ID": pmid, "error": str(e)}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            processed_ids.add(pmid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenRouter QA verification (serial/parallel processing).")
    parser.add_argument("--qa_pairs", type=str, required=True, help="Path to QA pairs JSONL file.")
    parser.add_argument("--abstracts", type=str, required=True, help="Path to abstracts JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Destination JSONL base file name for results. If multiple models are provided, the model name will be appended before the file extension.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["anthropic/claude-sonnet-4", "openai/chatgpt-4o-latest", "google/gemini-2.5-pro", "deepseek/deepseek-r1"],
        help="One or more OpenRouter model identifiers (space-separated)",
    )
    parser.add_argument(
        "--role_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "role.txt"),
        help="Path to the system role prompt file (defaults to role.txt in the script directory).",
    )
    parser.add_argument(
        "--api_keys_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.txt"),
        help="Path to the file containing API keys (one per line).",
    )
    parser.add_argument(
        "--n_process",
        type=int,
        default=1,
        help="Number of processes per API key.",
    )
    args = parser.parse_args()

    # Load datasets
    qa_pairs_raw = load_from_jsonl(args.qa_pairs)
    abstracts = load_from_json(args.abstracts)
    with open(args.role_file, "r") as f:
        role_prompt = f.read()
    qa_pairs = prepare_qa_pairs(qa_pairs_raw, abstracts)
    api_keys = read_api_keys(args.api_keys_file)
    n_api_keys = len(api_keys)
    n_process = args.n_process
    total_processes = n_api_keys * n_process
    def _safe(model_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_") or "model"
    base, ext = os.path.splitext(args.output_file)
    if not ext:
        ext = ".jsonl"
    for model_name in args.models:
        chunk_size = math.ceil(len(qa_pairs) / total_processes)
        processes = []
        for idx in range(total_processes):
            start = idx * chunk_size
            end = min((idx + 1) * chunk_size, len(qa_pairs))
            chunk = qa_pairs[start:end]
            if not chunk:
                continue
            api_key = api_keys[idx % n_api_keys]
            out_path = f"{base}_{_safe(model_name)}_part{idx}{ext}" if total_processes > 1 else (f"{base}_{_safe(model_name)}{ext}" if len(args.models) > 1 else args.output_file)
            p = Process(target=process_chunk, args=(chunk, model_name, role_prompt, out_path, api_key), name=f"Proc-{idx}")
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print(f"Processing complete for model {model_name}. Results saved to {base}_{_safe(model_name)}_part*.jsonl") 