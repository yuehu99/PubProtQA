"""LLM-based claim verification using OpenRouter (Gemini 2.5 Flash).

Replaces regex claim extraction + rule-based comparison with a single
LLM call per question category.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 3

_prompt_cache: dict[str, str] = {}


def _load_prompt(name: str) -> str:
    if name in _prompt_cache:
        return _prompt_cache[name]
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text()
    _prompt_cache[name] = text
    return text


def _load_system_prompt() -> str:
    return _load_prompt("_system")


CATEGORY_TO_PROMPT = {
    "protein_length": "protein_length",
    "molecular_weight": "molecular_weight",
    "active_site": "active_site",
    "signal_peptide": "signal_peptide",
    "ptm": "ptm",
    "subunit_structure": "subunit_structure",
    "subcellular_localization": "subcellular_localization",
    "domain_architecture": "domain_architecture",
    "function_ec": "function_ec",
    "species": "species",
    "secondary_structure": "secondary_structure",
    "ppi": "ppi",
}

CATEGORY_FACTS = {
    "protein_length": [
        "protein_length", "sequence_length_uniprot", "mature_protein_length",
        "signal_peptide", "transit_peptide",
    ],
    "molecular_weight": [
        "molecular_weight_da", "molecular_weight_kda",
        "mol_weight_uniprot", "signal_peptide",
    ],
    "active_site": ["active_sites", "keywords"],
    "signal_peptide": ["signal_peptide", "transit_peptide", "keywords"],
    "ptm": ["ptm_sites", "keywords", "function_text"],
    "subunit_structure": ["subunit_text", "keywords"],
    "subcellular_localization": ["subcellular_locations", "keywords"],
    "domain_architecture": ["domains", "interpro_domains", "keywords"],
    "function_ec": ["function_text", "ec_numbers", "keywords"],
    "species": ["organism"],
    "secondary_structure": ["secondary_structure", "keywords", "transmembrane"],
    "ppi": ["subunit_text", "organism", "accession"],
}


def _select_facts(facts: dict, category: str) -> dict:
    keys = CATEGORY_FACTS.get(category, list(facts.keys()))
    selected = {}
    for k in keys:
        v = facts.get(k)
        if v is not None and v != [] and v != "":
            selected[k] = v
    return selected


def _build_user_message(question: str, answer: str, facts: dict,
                        category: str) -> str:
    selected = _select_facts(facts, category)

    facts_str = json.dumps(selected, indent=2, default=str)

    return (
        f"QUESTION CATEGORY: {category}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"REFERENCE FACTS (from UniProt):\n{facts_str}"
    )


def call_openrouter(api_key: str, model: str, system_prompt: str,
                    user_prompt: str, temperature: float = DEFAULT_TEMPERATURE,
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    timeout: int = DEFAULT_TIMEOUT) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            rsp = r.json()
            if "error" in rsp:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"OpenRouter error: {rsp['error']}")
            return rsp
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError("Max retries exceeded")


def _parse_llm_response(content: str) -> dict:
    """Extract JSON from LLM response with fallback parsing."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r'\{[\s\S]*\}', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return {"claims": [], "aggregate": "not_applicable"}


def _normalize_verdicts(parsed: dict, category: str) -> list[dict]:
    """Convert LLM JSON output to the standard verdict dict format."""
    claims = parsed.get("claims", [])
    verdicts = []

    for claim in claims:
        status = claim.get("status", "not_applicable")
        if status not in ("verified", "contradicted", "not_applicable"):
            status = "not_applicable"

        verdict = {
            "claim_type": claim.get("claim_type", category),
            "claimed": claim.get("claimed_value", ""),
            "reference": None,
            "status": status,
            "question_category": category,
            "context": claim.get("context", ""),
            "reasoning": claim.get("reasoning", ""),
        }
        verdicts.append(verdict)

    return verdicts


def verify_category(question: str, answer: str, facts: dict,
                    category: str, api_key: str,
                    model: str = DEFAULT_MODEL) -> list[dict]:
    """Verify claims in a single category using the LLM.

    Returns list of verdict dicts.
    """
    prompt_name = CATEGORY_TO_PROMPT.get(category)
    if not prompt_name:
        return []

    system_text = _load_system_prompt()
    category_text = _load_prompt(prompt_name)
    system_prompt = system_text + "\n\n" + category_text

    user_prompt = _build_user_message(question, answer, facts, category)

    try:
        rsp = call_openrouter(api_key, model, system_prompt, user_prompt)
        content = (
            rsp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = _parse_llm_response(content)
        return _normalize_verdicts(parsed, category)
    except Exception:
        return []


def verify_qa_llm(question: str, answer: str, facts: dict,
                  api_key: str, model: str = DEFAULT_MODEL) -> list[dict]:
    """Full LLM-based verification for a QA pair.

    Classifies the question into categories, then calls the LLM for each.
    """
    from question_classifier import classify

    primary_cat, secondary_cats = classify(question, answer)

    if primary_cat == "other":
        return []

    categories_to_check = [primary_cat] + secondary_cats

    all_verdicts = []
    checked = set()
    for cat in categories_to_check:
        if cat in checked or cat not in CATEGORY_TO_PROMPT:
            continue
        checked.add(cat)
        verdicts = verify_category(question, answer, facts, cat,
                                   api_key=api_key, model=model)
        all_verdicts.extend(verdicts)

    return all_verdicts


def aggregate_label(checks: list[dict]) -> str:
    statuses = {c["status"] for c in checks}
    if "contradicted" in statuses:
        return "contradicted"
    if "verified" in statuses:
        return "verified"
    return "not_applicable"
