import os
import csv
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


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


# -------------------------------------------------------------------
# CSV I/O
# -------------------------------------------------------------------
def load_products_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load products from CSV. Required columns:
      - title
      - short_description
      - price
      - category
    """
    products: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"title", "short_description", "price", "category"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            row["title"] = (row.get("title") or "").strip()
            row["short_description"] = (row.get("short_description") or "").strip()
            row["category"] = (row.get("category") or "").strip()
            price_raw = (row.get("price") or "").strip()
            try:
                row["price"] = float(price_raw)
            except Exception:
                row["price"] = math.nan
            products.append(row)
    return products


def write_results_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "query",
        "rank",
        "similarity",
        "title",
        "short_description",
        "price",
        "category",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------
def get_embedding_serial(
    client: AzureOpenAI,
    deployment: str,
    text: str,
    max_retries: int = 5,
    retry_backoff_sec: float = 1.5,
) -> List[float]:
    """
    One embeddings.create() call per text, with retries/backoff on transient errors.
    """
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=deployment, input=text)
            return resp.data[0].embedding
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(retry_backoff_sec * (2 ** (attempt - 1)))


# -------------------------------------------------------------------
# Similarity
# -------------------------------------------------------------------
def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = np.asarray(list(vec_a), dtype=float)
    b = np.asarray(list(vec_b), dtype=float)
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0
    dist = cosine_distance(a, b)
    if np.isnan(dist):
        return 0.0
    return 1.0 - float(dist)


# -------------------------------------------------------------------
# Core flow
# -------------------------------------------------------------------
def ranking(
    input_csv: str,
    output_csv: str,
    queries: List[str],
    top_n: int = 3,
) -> None:
    cfg = load_config()
    client = get_azure_openai_client(cfg)

    products = load_products_csv(input_csv)
    if not products:
        print("No products found in input CSV.")
        return

    # Serial embeddings for products
    for p in products:
        p["_embedding"] = get_embedding_serial(client, cfg.deployment, p["short_description"])

    results_rows: List[Dict[str, Any]] = []

    # Serial embeddings for queries
    for query in queries:
        q_emb = get_embedding_serial(client, cfg.deployment, query)

        # Score
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for p in products:
            sim = cosine_similarity(q_emb, p["_embedding"])
            scored.append((sim, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_n]

        # Console output
        print(f"\nTop {top_n} matches for query: '{query}'")
        for rank, (sim, p) in enumerate(top, start=1):
            print(
                f"{rank}. [{sim:.4f}] {p['title']} — {p['short_description']} "
                f"(Category: {p['category']}, Price: ${p['price']:.2f})"
            )
            results_rows.append(
                {
                    "query": query,
                    "rank": rank,
                    "similarity": f"{sim:.6f}",
                    "title": p["title"],
                    "short_description": p["short_description"],
                    "price": p["price"],
                    "category": p["category"],
                }
            )

    write_results_csv(output_csv, results_rows)
    print(f"\nSaved results to: {output_csv}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic Search for Clothing Products (no batching)")
    parser.add_argument(
        "--input",
        default="clothing_products.csv",
        help="Path to products CSV (columns: title, short_description, price, category)",
    )
    parser.add_argument(
        "--output",
        default="search_results.csv",
        help="Path to write results CSV (default: search_results.csv)",
    )
    parser.add_argument(
        "--queries-file",
        default="queries.txt",
        help="Optional path to a text file with one query per line (no interactive input).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top matches per query to return (default: 3).",
    )

    return parser.parse_args()


def load_queries_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [q for q in lines if q]


if __name__ == "__main__":
    args = parse_args()

    queries_list = []
    if args.queries_file:
        queries_list = load_queries_from_file(args.queries_file)

    if len(queries_list) == 0:
        raise SystemExit("No queries provided (empty --queries-file?).")

    ranking(
        input_csv=args.input,
        output_csv=args.output,
        queries=queries_list,
        top_n=args.top_n,
    )
