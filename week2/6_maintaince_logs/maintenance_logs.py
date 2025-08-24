import os
import re
import sys
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI


@dataclass
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str


def load_config() -> AzureOpenAIConfig:
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview").strip()

    missing = [
        k
        for k, v in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY": api_key,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
        }.items()
        if not v
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nPlease set them (e.g., in a .env file) and re-run."
        )
    return AzureOpenAIConfig(endpoint, api_key, deployment, api_version)


def get_azure_openai_client(cfg: AzureOpenAIConfig) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=cfg.api_key,
        azure_endpoint=cfg.endpoint,
        api_version=cfg.api_version,
    )


ALLOWED_CATEGORIES: List[str] = [
    "Traffic",
    "Customer Issue",
    "Vehicle Issue",
    "Weather",
    "Sorting/Labeling Error",
    "Human Error",
    "Technical System Failure",
    "Other",
]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def build_classify_prompt(text: str) -> str:
    categories_bullets = "\n".join(f"- {c}" for c in ALLOWED_CATEGORIES)
    return f"""
You are a logistics assistant. Classify the following free-text incident log into exactly ONE category from this list:

{categories_bullets}

Decision rules:
- If the log says there was no delay or "arrived on time", return "Other".
- Use "Traffic" for congestion, accidents, road closures, detours, or construction.
- Use "Customer Issue" for recipient unavailable, incorrect address, refusal, unreachable.
- Use "Vehicle Issue" for engine/battery failures, breakdowns, flat tires, mechanical issues.
- Use "Weather" for storms, rain, hail, floods, extreme wind, etc.
- Use "Sorting/Labeling Error" for label/barcode/sorting problems.
- Use "Human Error" for wrong turns, routing mistakes, or operator mistakes.
- Use "Technical System Failure" for system/app/scanner/network/server issues.
- If nothing clearly matches, return "Other".

Log Entry:
\"\"\"{text}\"\"\"

Return ONLY the category name exactly as listed above.
""".strip()


def sanitize_category(raw: str) -> str:
    if not raw:
        return ""
    norm = normalize_text(raw)

    # Direct exact match
    for c in ALLOWED_CATEGORIES:
        if normalize_text(c) == norm:
            return c

    # Near-variants
    synonyms: Dict[str, str] = {
        "technical failure": "Technical System Failure",
        "technical system error": "Technical System Failure",
        "system failure": "Technical System Failure",
        "tech failure": "Technical System Failure",
        "sorting": "Sorting/Labeling Error",
        "labeling": "Sorting/Labeling Error",
        "sorting/labeling": "Sorting/Labeling Error",
        "human": "Human Error",
        "vehicle": "Vehicle Issue",
        "customer": "Customer Issue",
        "weather delay": "Weather",
        "traffic delay": "Traffic",
        "other delay": "Other",
    }
    if norm in synonyms:
        return synonyms[norm]

    # Conservative contains-based fallback
    contains_map = [
        ("traffic", "Traffic"),
        ("customer", "Customer Issue"),
        ("vehicle", "Vehicle Issue"),
        ("weather", "Weather"),
        ("label", "Sorting/Labeling Error"),
        ("sorting", "Sorting/Labeling Error"),
        ("human", "Human Error"),
        ("system", "Technical System Failure"),
        ("technical", "Technical System Failure"),
        ("other", "Other"),
    ]
    for key, val in contains_map:
        if key in norm:
            return val

    return ""


def classify_with_openai(client: AzureOpenAI, deployment: str, text: str) -> str:
    """
    Single-step LLM classification. On any API error or invalid output,
    returns 'Other' as a safe default.
    """
    prompt = build_classify_prompt(text)
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        final = sanitize_category(raw)
        return final if final in ALLOWED_CATEGORIES else "Other"
    except Exception as e:
        sys.stderr.write(f"[warn] classify_with_openai failed: {e}\n")
        return "Other"


SAMPLE_LOGS: List[Tuple[int, str]] = [
    (1,  "Driver reported heavy traffic on highway due to construction"),
    (2,  "Package not accepted, customer unavailable at given time"),
    (3,  "Vehicle engine failed during route, replacement dispatched"),
    (4,  "Unexpected rainstorm delayed loading at warehouse"),
    (5,  "Sorting label missing, required manual barcode scan"),
    (6,  "Driver took a wrong turn and had to reroute"),
    (7,  "No issue reported, arrived on time"),
    (8,  "Address was incorrect, customer unreachable"),
    (9,  "System glitch during check-in at loading dock"),
    (10, "Road accident caused a long halt near delivery point"),
]


def write_csv(fieldnames: List[str], rows: List[Dict[str, str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    cfg = load_config()
    client = get_azure_openai_client(cfg)

    fieldnames = ["log_id", "log_entry", "label"]
    results: List[Dict[str, str]] = []

    for log_id, entry in SAMPLE_LOGS:
        label = classify_with_openai(client, cfg.deployment, entry)
        results.append(
            {
                "log_id": str(log_id),
                "log_entry": entry,
                "label": label,
            }
        )

    out_csv = os.getenv("OUTPUT_CSV", "maintenance_logs.csv")
    write_csv(fieldnames, results, out_csv)
    print(f"Wrote {len(results)} rows to CSV: {out_csv}")


if __name__ == "__main__":
    main()
