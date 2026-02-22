import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

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


def chat_complete(
    client: AzureOpenAI,
    deployment: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> str:
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


SYSTEM_PROMPT = (
    "You are a helpful, professional Event Management Assistant. "
    "You help plan, host, and optimize networking events and conferences.\n\n"
    "IMPORTANT:\n"
    "- Use your internal reasoning to produce better answers, but NEVER reveal that reasoning or write out step-by-step thoughts.\n"
    "- Output only the final answers with concise, user-facing explanations.\n"
    "- If asked to 'show your reasoning', provide a brief rationale (1-2 sentences) instead of hidden chain-of-thought.\n"
    "- Keep answers structured and easy to scan."
)


FEW_SHOT_SENTIMENT: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Analyze the sentiment of this text: 'I love attending networking events!'",
    },
    {
        "role": "assistant",
        "content": (
            "Sentiment: Positive\n"
            "Confidence: 0.95\n"
            "Explanation: Expresses strong enjoyment and enthusiasm."
        ),
    },
    {
        "role": "user",
        "content": "Analyze the sentiment of this text: 'Networking can be really stressful sometimes.'",
    },
    {
        "role": "assistant",
        "content": (
            "Sentiment: Negative\n"
            "Confidence: 0.80\n"
            "Explanation: Describes stress and discomfort with networking."
        ),
    },
    {
        "role": "user",
        "content": "Analyze the sentiment of this text: 'The session was okay—some parts helped, some were dull.'",
    },
    {
        "role": "assistant",
        "content": (
            "Sentiment: Neutral\n"
            "Confidence: 0.70\n"
            "Explanation: Mixed reaction without strong positive or negative tone."
        ),
    },
]

# Guidance for safe reasoning-style outputs (no chain-of-thought leakage).
REASONING_STYLE_NOTE = (
    "Think through the problem privately. "
    "Return only:\n"
    "1) The answer in a clear, structured format.\n"
    "2) A very brief rationale (1-2 sentences)."
)


def mock_conversation_starters(client: AzureOpenAI, cfg: AzureOpenAIConfig) -> str:
    """
    Ask for conversation starters with brief why-it-works. Demonstrates role usage and
    reasoning-friendly prompts without revealing chain-of-thought.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "We're hosting a cloud-computing networking mixer after the keynote.",
        },
        {
            "role": "assistant",
            "content": "Noted. I can help with conversation starters and flows.",
        },
        {
            "role": "user",
            "content": (
                f"{REASONING_STYLE_NOTE}\n\n"
                "Give 8 friendly conversation starters tailored to cloud professionals.\n"
                "- Each item: a short opener + a 'Why it works' phrase (≤ 10 words).\n"
                "- Avoid clichés. Encourage inclusive, non-awkward phrasing.\n"
                "- Output as a numbered list."
            ),
        },
    ]

    reply = chat_complete(
        client, cfg.deployment, messages, temperature=0.6, max_tokens=700
    )
    return reply


SENTIMENT_INPUTS: List[str] = [
    "I met so many inspiring people—this event gave me a boost!",
    "The venue's acoustics made it hard to hear. Kinda frustrating.",
    "Met a few contacts. Overall fine, but nothing stood out.",
]


def sentiment_messages_with_few_shots(text: str) -> List[Dict[str, str]]:
    """
    Build a conversation that includes the system prompt + few-shot examples,
    then asks for the new sentiment classification in a structured, CoT-safe way.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_SENTIMENT)
    messages.append(
        {
            "role": "user",
            "content": (
                f"{REASONING_STYLE_NOTE}\n\n"
                f"Analyze the sentiment of this text: '{text}'\n"
                "- Return exactly these fields in plain text:\n"
                "  Sentiment: <Positive|Neutral|Negative>\n"
                "  Confidence: <0.00-1.00>\n"
                "  Explanation: <concise reason (≤ 20 words)>\n"
            ),
        }
    )
    return messages


def mock_sentiment_batch(
    client: AzureOpenAI, cfg: AzureOpenAIConfig
) -> List[Tuple[str, str]]:
    """
    Classify multiple inputs. Returns list of (input_text, model_reply).
    """
    results: List[Tuple[str, str]] = []
    for text in SENTIMENT_INPUTS:
        reply = chat_complete(
            client,
            cfg.deployment,
            sentiment_messages_with_few_shots(text),
            temperature=0.2,
            max_tokens=250,
        )
        results.append((text, reply))
    return results


def mock_context_reasoning(client: AzureOpenAI, cfg: AzureOpenAIConfig) -> str:
    """
    Provide an actionable, context-aware operations plan (e.g., keynote delay scenario).
    We encourage reasoning but request only final steps + brief justification.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "We have 350 attendees in the main hall after lunch.",
        },
        {
            "role": "assistant",
            "content": "Understood. Crowd energy may dip post-lunch—let’s keep them engaged.",
        },
        {
            "role": "user",
            "content": (
                f"{REASONING_STYLE_NOTE}\n\n"
                "Scenario: The keynote speaker is delayed by 15–20 minutes. "
                "Design a calm, professional micro-agenda to keep attendees engaged.\n"
                "Constraints:\n"
                "- A/V crew available\n"
                "- Two volunteer MCs\n"
                "- Sponsor booths in foyer\n"
                "- No food service inside the hall\n\n"
                "Output format:\n"
                "Title\n"
                "Duration: <mins>\n"
                "Steps: (numbered, timeboxed, specific)\n"
                "Roles: (who does what)\n"
                "Comms: (exact 1-2 announcement lines)\n"
                "Rationale (brief): (≤ 2 sentences)"
            ),
        },
    ]

    reply = chat_complete(
        client, cfg.deployment, messages, temperature=0.6, max_tokens=600
    )
    return reply


def print_header(title: str):
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_block(text: str):
    print(text.strip())
    print("\n")


def try_extract_fields_block(raw: str) -> Dict[str, str]:
    """
    Helper that attempts to parse a simple key: value block for Sentiment/Confidence/Explanation.
    Falls back to raw if parsing fails.
    """
    out = {"Sentiment": "", "Confidence": "", "Explanation": ""}
    try:
        # Normalize newlines, then parse lines like "Key: Value"
        for line in raw.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                val = val.strip()
                if key == "sentiment":
                    out["Sentiment"] = val
                elif key == "confidence":
                    out["Confidence"] = val
                elif key == "explanation":
                    out["Explanation"] = val

        if out["Sentiment"]:
            return out
    except Exception:
        pass
    return {"Raw": raw}


def print_sentiment_results(pairs: List[Tuple[str, str]]):
    print_header("B) Sentiment Analysis (Few-Shot + CoT-Safe Reasoning)")
    for i, (text, reply) in enumerate(pairs, start=1):
        parsed = try_extract_fields_block(reply)
        print(f"[Sample {i}] Text: {text}")
        if "Raw" in parsed:
            print("  Model Output (raw):")
            print_block("  " + parsed["Raw"].replace("\n", "\n  "))
        else:
            print("  Result:")
            print(f"    Sentiment  : {parsed['Sentiment']}")
            print(f"    Confidence : {parsed['Confidence']}")
            print(f"    Explanation: {parsed['Explanation']}\n")


def main():
    cfg = load_config()
    client = get_azure_openai_client(cfg)

    # A) Conversation starters
    starters = mock_conversation_starters(client, cfg)
    print_header("A) Conversation Starters for a Cloud Networking Mixer")
    print_block(starters)

    # B) Sentiment analysis (batch, 3 samples)
    sentiment_pairs = mock_sentiment_batch(client, cfg)
    print_sentiment_results(sentiment_pairs)

    # C) Context-aware plan for a keynote delay
    ops_plan = mock_context_reasoning(client, cfg)
    print_header("C) Micro-Agenda Plan: Keynote Delay Handling")
    print_block(ops_plan)


if __name__ == "__main__":
    main()
