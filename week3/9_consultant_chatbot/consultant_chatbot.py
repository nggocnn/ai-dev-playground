import os
import csv
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI

# -------------------- ENV --------------------
# Expect these via .env or environment variables
# Embeddings
#   AZURE_OPENAI_EMBEDDING_API_KEY
#   AZURE_OPENAI_EMBEDDING_ENDPOINT
#   AZURE_OPENAI_EMBED_MODEL              (deployment name)
#
# Chat LLM
#   AZURE_OPENAI_LLM_API_KEY
#   AZURE_OPENAI_LLM_ENDPOINT
#   AZURE_OPENAI_LLM_MODEL                (deployment name)
#
# AZURE_OPENAI_API_VERSION          (default: 2024-07-01-preview)
#
# Chroma persistence (optional; can override by CLI):
#   CHROMA_DIR (default: ./chroma)
#   CHROMA_COLLECTION_NAME (default: laptops)

load_dotenv()

# -------------------- CONFIG --------------------
@dataclass
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str

def get_azure_openai_client(cfg: AzureOpenAIConfig) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=cfg.api_key,
        azure_endpoint=cfg.endpoint,
        api_version=cfg.api_version,
    )

# Embedding client config
EMBEDDING_CFG = AzureOpenAIConfig(
    endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", ""),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", ""),
    deployment=os.getenv("AZURE_OPENAI_EMBED_MODEL", ""),  # deployment name
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview"),
)

# LLM client config
LLM_CFG = AzureOpenAIConfig(
    endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT", ""),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY", ""),
    deployment=os.getenv("AZURE_OPENAI_LLM_MODEL", ""),  # deployment name
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview"),
)

# -------------------- Clients --------------------
embedding_client = get_azure_openai_client(EMBEDDING_CFG)
llm_client = get_azure_openai_client(LLM_CFG)

# -------------------- Core helpers --------------------
def require_env():
    missing = [
        name for name, val in [
            ("AZURE_OPENAI_EMBEDDING_API_KEY", EMBEDDING_CFG.api_key),
            ("AZURE_OPENAI_EMBEDDING_ENDPOINT", EMBEDDING_CFG.endpoint),
            ("AZURE_OPENAI_EMBED_MODEL", EMBEDDING_CFG.deployment),
            ("AZURE_OPENAI_LLM_API_KEY", LLM_CFG.api_key),
            ("AZURE_OPENAI_LLM_ENDPOINT", LLM_CFG.endpoint),
            ("AZURE_OPENAI_LLM_MODEL", LLM_CFG.deployment),
        ] if not val
    ]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))

def get_embedding(text: str, max_retries: int = 3, sleep_s: float = 1.0) -> List[float]:
    text = (text or "").strip()
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = embedding_client.embeddings.create(
                model=EMBEDDING_CFG.deployment,
                input=text,
            )
            return r.data[0].embedding
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(sleep_s * attempt)
            else:
                raise RuntimeError(f"Embedding failed after {max_retries} attempts: {e}") from e
    raise last_err  # pragma: no cover

SYSTEM_PROMPT = (
    "You are a laptop recommendation specialist.\n"
    "You will receive a user requirement and a CONTEXT containing only the top relevant laptops.\n"
    "CRITICAL RULES:\n"
    "1) Only recommend from the provided CONTEXT. Do NOT invent models/specs/prices.\n"
    "2) If information is missing in CONTEXT, label it as 'unknown'.\n"
    "3) Focus on the user's stated needs (gaming, business/office, student/general, portability, battery, OS, RAM, storage, GPU, display).\n"
    "4) Provide 1-3 best-fit options with brief pros/cons + why they match.\n"
    "5) Be concise and helpful. End with a one-line 'who should pick this' summary for each pick.\n"
)

def ask_llm(context: str, user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"USER REQUIREMENTS:\n{user_input}\n\n"
            f"CONTEXT (Top relevant laptops):\n{context}\n\n"
            "Return a short answer in Markdown with:\n"
            "- A brief opener (1-2 sentences)\n"
            "- 1-3 recommendations (Name, key specs from context, pros/cons, why it fits)\n"
            "- A short closing tip (e.g., what to double-check)\n"
        }
    ]
    r = llm_client.chat.completions.create(
        model=LLM_CFG.deployment,
        messages=messages,
        temperature=0.3,
    )
    return r.choices[0].message.content

# -------------------- Data I/O --------------------
def load_laptops_csv_strict(csv_path: str) -> List[Dict]:
    """
    Strict loader: expects EXACT headers: id,name,description,tags
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = ["id", "name", "description", "tags"]
        if [h.strip() for h in (reader.fieldnames or [])] != expected:
            raise ValueError(f"CSV must have headers exactly: {','.join(expected)}")
        rows = []
        for row in reader:
            rows.append({
                "id": row["id"].strip(),
                "name": row["name"].strip(),
                "description": row["description"].strip(),
                "tags": row["tags"].strip(),
            })
        return rows

def load_queries_txt(txt_path: str) -> List[str]:
    qs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            qs.append(s)
    return qs

# -------------------- Chroma helpers --------------------
def get_chroma_collection(chroma_dir: str, collection_name: str, create_if_missing: bool = True):
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(allow_reset=True))
    try:
        col = client.get_collection(collection_name)
    except Exception:
        if not create_if_missing:
            raise
        col = client.create_collection(collection_name)
    return col

def format_doc_for_embedding(description: str, tags: str) -> str:
    # Richer text improves retrieval
    return f"Description: {description}\nTags: {tags}"

def build_context_from_results(results: Dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    lines = []
    for idx, (doc, meta, _id) in enumerate(zip(docs, metas, ids), start=1):
        lines.append(
            f"{idx}. Name: {meta.get('name','unknown')} (ID: {_id})\n"
            f"   Description: {doc}\n"
            f"   Tags: {meta.get('tags','')}"
        )
    return "\n\n".join(lines)

# -------------------- Modes --------------------
def mode_embed(csv_path: str, chroma_dir: str, collection_name: str, reset: bool, batch: int = 16):
    client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(allow_reset=True))
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[embed] Deleted existing collection '{collection_name}'.")
        except Exception:
            pass
    try:
        collection = client.get_collection(collection_name)
        print(f"[embed] Using existing collection '{collection_name}'.")
    except Exception:
        collection = client.create_collection(collection_name)
        print(f"[embed] Created collection '{collection_name}'.")

    laptops = load_laptops_csv_strict(csv_path)
    print(f"[embed] Loaded {len(laptops)} laptops from {csv_path}")

    ids, embs, docs, metas = [], [], [], []

    def flush():
        nonlocal ids, embs, docs, metas
        if not ids:
            return
        try:
            # Try add first (faster). If duplicates exist, fall back to update per-id.
            collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        except Exception:
            # Fallback: update each id (works if they already exist)
            for _id, _e, _d, _m in zip(ids, embs, docs, metas):
                try:
                    collection.update(ids=[_id], embeddings=[_e], documents=[_d], metadatas=[_m])
                except Exception:
                    # If update fails (id not found), try add for that one
                    try:
                        collection.add(ids=[_id], embeddings=[_e], documents=[_d], metadatas=[_m])
                    except Exception as e:
                        print(f"[embed][warn] Failed upsert for id={_id}: {e}")
        ids, embs, docs, metas = [], [], [], []

    for i, lp in enumerate(laptops, start=1):
        combined_text = format_doc_for_embedding(lp["description"], lp["tags"])
        vec = get_embedding(combined_text)
        ids.append(lp["id"])
        embs.append(vec)
        docs.append(combined_text)
        metas.append({"name": lp["name"], "tags": lp["tags"]})
        if len(ids) >= batch:
            flush()
    flush()
    print(f"[embed] Index build complete. Items in collection: {collection.count()}")

def mode_query(queries_path: str, chroma_dir: str, collection_name: str, n_results: int):
    collection = get_chroma_collection(chroma_dir, collection_name, create_if_missing=False)
    queries = load_queries_txt(queries_path)
    if not queries:
        raise RuntimeError(f"No queries found in {queries_path}")

    print("\n" + "=" * 80)
    print("Laptop Consultant — Recommendations")
    print("=" * 80 + "\n")

    for q in queries:
        print("-" * 80)
        print(f"User input: {q}")

        q_vec = get_embedding(q)
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        context = build_context_from_results(results)
        answer = ask_llm(context, q)
        print("\nLLM Recommendation:\n")
        print(answer)
        print("-" * 80 + "\n")

# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Laptop Consultant Chatbot (Azure OpenAI + ChromaDB)"
    )
    parser.add_argument(
        "--mode",
        choices=["embed", "query"],
        default="query",
        help="Flow mode. 'embed' builds/updates the index from CSV; 'query' answers from queries file (default).",
    )
    parser.add_argument(
        "--laptops",
        default="./laptops.csv",
        help="Path to laptops CSV (id,name,description,tags). Used in --mode=embed.",
    )
    parser.add_argument(
        "--queries",
        default="./queries.txt",
        help="Path to queries text file. Used in --mode=query.",
    )
    parser.add_argument(
        "--n_results",
        type=int,
        default=3,
        help="Top N results to retrieve from Chroma in query mode.",
    )
    parser.add_argument(
        "--chroma_dir",
        default=os.getenv("CHROMA_DIR", "./chroma"),
        help="Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION_NAME", "laptops"),
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="(embed mode) Delete and recreate the collection before indexing.",
    )
    return parser.parse_args()

def main():
    require_env()
    args = parse_args()
    if args.mode == "embed":
        if not os.path.exists(args.laptops):
            raise FileNotFoundError(f"CSV not found: {args.laptops}")
        mode_embed(
            csv_path=args.laptops,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            reset=args.reset,
        )
    else:
        if not os.path.exists(args.queries):
            raise FileNotFoundError(f"Queries file not found: {args.queries}")
        mode_query(
            queries_path=args.queries,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            n_results=args.n_results,
        )

if __name__ == "__main__":
    main()
