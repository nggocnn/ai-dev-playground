import os
import json
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import (
    AzureOpenAI,
    APIError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


CLIENT: Optional[AzureOpenAI] = None
DEPLOYMENT: str = ""


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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_itinerary",
            "description": "Return a complete travel itinerary in structured JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "City or country to visit.",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of days in the itinerary.",
                    },
                    "currency": {
                        "type": "string",
                        "description": "3-letter currency code for budget estimates (e.g., USD, EUR).",
                    },
                    "travel_style": {
                        "type": "string",
                        "description": "Optional style: budget, mid-range, luxury, family, foodie, etc.",
                    },
                    "highlights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Top highlights to expect from the trip.",
                    },
                    "daily_plan": {
                        "type": "array",
                        "description": "Day-by-day plan with time-of-day activities.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day": {"type": "integer"},
                                "morning": {"type": "string"},
                                "afternoon": {"type": "string"},
                                "evening": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                            "required": ["day", "morning", "afternoon", "evening"],
                        },
                    },
                    "estimated_budget": {
                        "type": "object",
                        "description": "Rough budget per day and total.",
                        "properties": {
                            "per_day": {"type": "number"},
                            "total": {"type": "number"},
                            "breakdown": {
                                "type": "object",
                                "properties": {
                                    "lodging": {"type": "number"},
                                    "food": {"type": "number"},
                                    "transport": {"type": "number"},
                                    "activities": {"type": "number"},
                                    "misc": {"type": "number"},
                                },
                            },
                        },
                    },
                    "tips": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Helpful travel tips for the destination.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the itinerary and who it suits best.",
                    },
                },
                "required": ["destination", "days", "daily_plan"],
            },
        },
    }
]

SYSTEM_INSTRUCTION = (
    "You are a travel planner. Produce a practical, concise itinerary for the given destination and number of days. "
    "You MUST return the result by calling the function submit_itinerary with a valid JSON conforming to its schema. "
    "Keep descriptions clear and helpful."
)

TransientErrors = (RateLimitError, APIError, APIConnectionError, APITimeoutError)


@retry(
    retry=retry_if_exception_type(TransientErrors),
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _invoke_chat_completion(messages: List[Dict[str, str]]) -> Any:
    if CLIENT is None or not DEPLOYMENT:
        raise RuntimeError("Client not initialized.")
    return CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "submit_itinerary"}},
        temperature=0.7,
    )


def _parse_tool_args(response: Any) -> Dict[str, Any]:
    try:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls or tool_calls[0].type != "function":
            raise ValueError("Model response did not include a function tool call.")
        args_json = tool_calls[0].function.arguments or "{}"
        return json.loads(args_json)
    except (IndexError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse tool arguments: {exc}") from exc


def _build_messages(
    prompt: str, destination: str, days: int, travel_style: Optional[str] = None
) -> List[Dict[str, str]]:
    user_content = (
        f"{prompt}\n\n"
        f"Destination: {destination}\n"
        f"Days: {days}\n"
        + (f"Travel style: {travel_style}\n" if travel_style else "")
        + "Use realistic attractions and time blocks; keep each day balanced."
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def normalize_result(
    tool_args: Dict[str, Any], destination: str, days: int, travel_style: Optional[str]
) -> Dict[str, Any]:
    out = dict(tool_args) if isinstance(tool_args, dict) else {}
    out.setdefault("destination", destination)
    out.setdefault("days", days)
    if travel_style and "travel_style" not in out:
        out["travel_style"] = travel_style
    return out


def call_openai_function(
    prompt: str, destination: str, days: int, travel_style: Optional[str] = None
) -> Dict[str, Any]:
    """
    Single item call:
    - Builds messages
    - Calls Azure OpenAI with retries
    - Parses the tool call JSON
    - Normalizes/returns a structured dict
    """
    messages = _build_messages(prompt, destination, days, travel_style)
    logger.debug("Sending request for %s (%d days)", destination, days)
    response = _invoke_chat_completion(messages)
    tool_args = _parse_tool_args(response)
    result = normalize_result(tool_args, destination, days, travel_style)
    return result


def batch_process(
    inputs: List[Dict[str, Any]], *, max_workers: int = 2, sleep_between: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Process a list of inputs with limited concurrency. Order is preserved.
    Returns a list of results (dict). If an item fails, an error dict is returned for that slot.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(inputs)

    def _task(idx: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        destination = item.get("destination", "Unknown")
        try:
            res = call_openai_function(
                prompt=item["prompt"],
                destination=destination,
                days=int(item["days"]),
                travel_style=item.get("travel_style"),
            )
            return idx, res
        except TransientErrors as te:
            logger.error(
                "Transient error (max retries hit) for %s: %s", destination, te
            )
            return idx, {
                "error": f"Transient error after retries: {type(te).__name__}",
                "destination": destination,
            }
        except Exception as e:
            logger.exception("Non-transient error for %s", destination)
            return idx, {"error": str(e), "destination": destination}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_task, idx, item): idx for idx, item in enumerate(inputs)
        }
        for future in as_completed(future_to_idx):
            idx, value = future.result()
            results[idx] = value
            if sleep_between > 0:
                time.sleep(sleep_between)

    return [r if r is not None else {"error": "Unknown failure"} for r in results]


if __name__ == "__main__":
    cfg = load_config()
    CLIENT = get_azure_openai_client(cfg)
    DEPLOYMENT = cfg.deployment

    batch_inputs = [
        {
            "prompt": "Plan a balanced, must-see travel itinerary.",
            "destination": "Paris",
            "days": 3,
            "travel_style": "mid-range",
        },
        {
            "prompt": "Plan a foodie-forward, neighborhood-centric trip.",
            "destination": "Tokyo",
            "days": 5,
            "travel_style": "foodie",
        },
        {
            "prompt": "Plan a mix of museums and iconic sights.",
            "destination": "New York",
            "days": 4,
        },
    ]

    logger.info("Starting batch of %d items (workers=%d)", len(batch_inputs), 2)
    outputs = batch_process(batch_inputs, max_workers=2, sleep_between=0.2)

    for idx, out in enumerate(outputs):
        header = f"Result for {batch_inputs[idx]['destination']} ({batch_inputs[idx]['days']} days):"
        print("\n" + header)
        print("-" * len(header))
        if "error" in out:
            print(f"ERROR: {out['error']}")
            continue
        summary = {
            "destination": out.get("destination"),
            "days": out.get("days"),
            "highlights": out.get("highlights", [])[:3],
            "budget_total": out.get("estimated_budget", {}).get("total"),
            "day1": out.get("daily_plan", [{}])[0],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
