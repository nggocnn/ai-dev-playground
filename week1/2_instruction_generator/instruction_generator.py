import os
import csv
import argparse

from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv
from openai import AzureOpenAI


SYSTEM_PROMPT = """You are an expert automotive manufacturing supervisor.
Generate detailed, step-by-step WORK INSTRUCTIONS for a new-model production task.
Write for assembly/quality/technician staff. Be concise, unambiguous, and action oriented.

MANDATORY CONTENT:
1) Safety: list PPE, lockout/tagout, ESD/voltage, chemical or lifting hazards.
2) Tools & Materials: exact tools (e.g., calibrated torque wrench 10-60 N·m), fixtures, targets, software.
3) Preparation: relevant pre-checks or line conditions; part and revision verification.
4) Procedure: numbered steps with clear verbs; include key parameters (torque, angles, distances, pressures, times).
5) Acceptance Criteria & Verification: measurable checks, test pass/fail thresholds, logging/traceability.
6) Nonconformance: what to do if a check fails.
7) Documentation: what and where to record (MES/Traveler/Log).

STYLE:
- Numbered steps, short sentences, SI units, use bold labels like **Safety** when helpful.
- Do NOT invent values if not given. If a spec is unknown, write "per spec XXXX" and leave a placeholder like "<torque spec>".
- Keep to production-ready clarity.
"""

USER_PROMPT_TEMPLATE = """Task:
\"\"\"{task}\"\"\"

Produce WORK INSTRUCTIONS.
"""


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
            + ". Put them in a .env file or your shell environment. See README."
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


def generate_instruction(client: AzureOpenAI, task: str, deployment: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task=task)},
    ]

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            n=1,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Empty response content from Azure OpenAI.")
        return content
    except Exception as e:
        print(f"Error: {str(e)}")


def read_task(input_csv: str) -> List[Tuple[str, str]]:
    tasks: List[Tuple[str, str]] = []
    with open(input_csv, "r", newline="", encoding="utf=8") as f:
        reader = csv.DictReader(f)
        field_map = {(k or "").strip().lower(): k for k in reader.fieldnames or []}
        id_key = field_map.get("id")
        task_key = field_map.get("task_description")

        for row in reader:
            task_id = (
                (row.get(id_key) if id_key else "").strip() if row.get(id_key) else ""
            )
            task_desc = (row.get(task_key) or "").strip()

            if task_desc:
                tasks.append((task_id, task_desc))

    if not tasks:
        raise RuntimeError(f"No tasks found in {input_csv}")

    return tasks


def write_output(output_csv: str, row: Tuple[str, str], header_written: bool) -> None:
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not header_written:
            writer.writerow(["Task Description Generated", "Work Instruction"])
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Generate work instructions from task descriptions using Azure OpenAI."
    )
    parser.add_argument(
        "--input",
        default="./tasks.csv",
        help="Path to input CSV (default: data/tasks.csv)",
    )
    parser.add_argument(
        "--output",
        default="./instructions.csv",
        help="Path to output CSV (default: out/instructions.csv)",
    )
    args = parser.parse_args()

    azure_openai_cfg = load_config()
    azure_openai_client = get_azure_openai_client(azure_openai_cfg)

    tasks = read_task(args.input)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    header_written = False
    for task_id, task_desc in tasks:
        print(f"Generating instructions for task id={task_id}...")
        instruction = generate_instruction(
            azure_openai_client, task_desc, azure_openai_cfg.deployment
        )
        write_output(args.output, (task_desc, instruction), header_written)
        header_written = True


if __name__ == "__main__":
    main()
