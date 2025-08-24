import os
import sys
import glob
import argparse

from dataclasses import dataclass
from typing import List
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
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    missing = [
        k
        for k, v in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ("AZURE_OPENAI_API_VERSION", api_version),
        ]
        if not v
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Put them in a .env file or your shell environment."
        )
    return AzureOpenAIConfig(
        endpoint=endpoint,
        api_key=api_key,
        deployment=deployment,
        api_version=api_version,
    )


def get_azure_openai_client(cfg: AzureOpenAIConfig) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=cfg.endpoint,
        api_key=cfg.api_key,
        api_version=cfg.api_version,
    )


SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in summarizing meeting notes. "
    "Produce concise, structured summaries with sections: Key Points, Decisions, Action Items. "
    "Prefer bullets, be specific and actionable. If owners/dates are present, include them. "
    "Do not invent details; if information is missing or incomplete, say so briefly."
)

CHUNK_CONTEXT_NOTE = (
    "Context: This is one segment of a longer meeting transcript. It may refer to earlier or later "
    "parts not shown here. Summarize this segment independently. If references seem incomplete "
    "(e.g., missing owners/dates), note them briefly without inventing details."
)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def chunk_text_with_overlap(text: str, limit: int, overlap_ratio: float) -> List[str]:
    lines = text.splitlines()
    n = len(lines)
    chunks = []
    i = 0
    while i < n:
        count = 0
        j = i
        buf = []
        while j < n:
            ln = lines[j] + "\n"
            if buf and (count + len(ln) > limit):
                break
            buf.append(lines[j])
            count += len(ln)
            j += 1
        if not buf:
            break
        chunk = "\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        overlap_lines = max(1, int(len(buf) * overlap_ratio))
        i = max(i + len(buf) - overlap_lines, i + 1)
    return chunks or [text]


def summarize_chunk(
    client: AzureOpenAI, deployment: str, chunk: str, max_tokens: int = 900
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{CHUNK_CONTEXT_NOTE}\n\n"
                "Summarize the following meeting transcript segment. Extract:\n"
                "• Key Points\n• Decisions\n• Action Items (with owners & due dates if mentioned)\n\n"
                f"Transcript segment:\n{chunk}"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def synthesize_summary(
    client: AzureOpenAI, deployment: str, partials: List[str], max_tokens: int = 1200
) -> str:
    merged = "\n\n---\n\n".join(partials)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are given multiple partial summaries from one long meeting.\n"
                "Merge them into a single, concise, non-redundant summary with exactly these sections:\n"
                "## Key Points\n## Decisions\n## Action Items\n"
                "Unify duplicates, keep bullets crisp and actionable. Do not invent details.\n\n"
                f"Partial summaries:\n{merged}"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def summarize_text(
    client: AzureOpenAI,
    deployment: str,
    text: str,
    chunk_char_limit: int,
    overlap_ratio: float,
    use_chunking: bool,
) -> str:
    if not use_chunking:
        return summarize_chunk(client, deployment, text)

    chunks = chunk_text_with_overlap(text, chunk_char_limit, overlap_ratio)
    if len(chunks) == 1:
        return summarize_chunk(client, deployment, chunks[0])
    partials = [summarize_chunk(client, deployment, ch) for ch in chunks]
    return synthesize_summary(client, deployment, partials)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Meeting Summarizer (Azure OpenAI)"
    )
    parser.add_argument(
        "--transcripts-dir",
        default="./inputs",
        help="Folder with .txt meeting transcripts (recursively searched). Default: ./inputs",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Folder to save summary files. Default: ./outputs",
    )
    parser.add_argument(
        "--chunk-char-limit",
        type=int,
        default=12000,
        help="Approximate max characters per chunk before splitting. Default: 12000",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.15,
        help="Fraction of previous chunk's lines to overlap into next (0.0-0.4). Default: 0.15",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable chunking — send the entire transcript in one request",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    client = get_azure_openai_client(cfg)

    txt_paths = sorted(
        glob.glob(os.path.join(args.transcripts_dir, "**", "*.txt"), recursive=True)
    )
    if not txt_paths:
        print(f"[ERROR] No .txt files found in {args.transcripts_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Found {len(txt_paths)} transcript(s) in {args.transcripts_dir}")

    for path in txt_paths:
        print(f"[INFO] Summarizing: {path}")
        try:
            text = read_text(path)
            summary = summarize_text(
                client,
                cfg.deployment,
                text,
                args.chunk_char_limit,
                args.overlap_ratio,
                use_chunking=not args.no_chunk,
            )
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.output_dir, f"{base}_summary.txt")
            write_text(out_path, summary)
            print(f"[OK] Saved summary -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {path}: {e}")

    print(f"[DONE] All summaries saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
