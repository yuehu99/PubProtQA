import os
import numpy
import requests
import json
import argparse
import re
from tqdm import tqdm  # Progress bar for serial processing

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

# NOTE: The earlier batch-based helper is removed. The following convenience
# function now simply filters QA pairs and enriches them with abstracts so we
# can iterate over them serially.


def prepare_qa_pairs(qa_pairs, articles):
    """Filter QA pairs, attach abstracts, and return the cleaned list."""

    # Lookup of PubMed_ID → abstract
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


def get_all_processed_ids(output_dir):
    """Aggregate PubMed_IDs already processed across all output shards."""
    processed_ids = set()
    if not os.path.exists(output_dir):
        return processed_ids

    for fname in os.listdir(output_dir):
        if fname.startswith("results_process_") and fname.endswith(".jsonl"):
            with open(os.path.join(output_dir, fname), "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        processed_ids.add(record["PubMed_ID"])
                    except Exception:
                        continue
    return processed_ids


# ---------------------------------------------
# OpenRouter wrapper (existing class stays untouched)
# ---------------------------------------------

class OpenRouter:
    """
    This class is used to interact with the OpenRouter API.

    NOTE: specify API key as an os environment variable. 
        1. "export OPENROUTER_APIKEY=<your_api_key>" to create the environment variable
        3. "echo $OPENROUTER_APIKEY" to check if the environment variable is set

    To see list of all available models, see: https://openrouter.ai/models
        - see specific api instructions for each model to get the specific model name string
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

def process_batch(batch_idx: int, input_file: str, output_dir: str, model: str, role: str):
    """Process a single batch file and write results to *output_dir*."""

    router = OpenRouter()
    processed_ids = get_all_processed_ids(output_dir)  # global checkpoint

    # Load batch
    with open(input_file, "r") as f:
        batch = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_process_{batch_idx}.jsonl")

    with open(output_file, "a") as out_f:
        for qa_entry in tqdm(batch, desc=f"Batch {batch_idx}"):
            pmid = qa_entry["PubMed_ID"]
            if pmid in processed_ids:
                continue  # Skip already processed globally

            abstract = qa_entry.get("abstract", "")
            proteins = qa_entry.get("proteins", [])
            if not proteins or not any(p.get("conversations") for p in proteins):
                continue

            prompt = build_prompt(abstract, proteins)
 
            try:
                # Compose an OpenRouter compliant payload so we can incorporate the custom system *role*
                url = "https://openrouter.ai/api/v1/chat/completions"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": role},
                        {"role": "user", "content": prompt},
                    ],
                }

                response = requests.post(
                    url=url,
                    headers=router.headers,
                    data=json.dumps(payload),
                )

                response_json = response.json()

                if "error" in response_json:
                    raise RuntimeError(response_json["error"])

                content = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "No content")
                )

                record = {"PubMed_ID": pmid, "response": content}
            except Exception as e:
                record = {"PubMed_ID": pmid, "error": str(e)}

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            processed_ids.add(pmid)  # update global checkpoint set


# ---------------------------------------------
# CLI
# ---------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run OpenRouter QA verification (serial processing).")

    parser.add_argument("--qa_pairs", type=str, required=True, help="Path to QA pairs JSONL file.")
    parser.add_argument("--abstracts", type=str, required=True, help="Path to abstracts JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Destination JSONL base file name for results. If multiple models are provided, the model name will be appended before the file extension.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["anthropic/claude-sonnet-4", "openai/chatgpt-4o-latest", "google/gemini-2.5-pro", "deepseek/deepseek-r1"],
        help="One or more OpenRouter model identifiers (space-separated)",
    )
    
    # Default to role.txt in the same directory as this script
    default_role_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "role.txt")
    parser.add_argument(
        "--role_file",
        type=str,
        default=default_role_path,
        help="Path to the system role prompt file (defaults to role.txt in the script directory).",
    )

    args = parser.parse_args()

    # Load datasets
    qa_pairs_raw = load_from_jsonl(args.qa_pairs)
    abstracts = load_from_json(args.abstracts)

    with open(args.role_file, "r") as f:
        role_prompt = f.read()

    qa_pairs = prepare_qa_pairs(qa_pairs_raw, abstracts)

    router = OpenRouter()

    # helper to sanitize model name for filenames
    def _safe(model_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_") or "model"

    base, ext = os.path.splitext(args.output_file)
    if not ext:
        ext = ".jsonl"

    for model_name in args.models:
        out_path = (
            f"{base}_{_safe(model_name)}{ext}" if len(args.models) > 1 else args.output_file
        )

        processed_ids = set()
        if os.path.exists(out_path):
            with open(out_path, "r") as f_res:
                for line in f_res:
                    try:
                        processed_ids.add(json.loads(line)["PubMed_ID"])
                    except Exception:
                        continue

        with open(out_path, "a") as out_f:
            for qa_entry in tqdm(qa_pairs, desc=f"Processing {model_name}"):
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

                    # Store both the extracted assistant reply *and* the full raw response
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

        print(f"Processing complete for model {model_name}. Results saved to {out_path}")
