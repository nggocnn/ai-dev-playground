# Efficient Azure OpenAI API Usage with Function Calling, Batching, and Robust Retries

This README explains how to run and extend the provided Python script that demonstrates **Azure OpenAI** usage with:

* **Function Calling (tools)** to enforce structured JSON outputs
* **Batching** with limited **concurrency** (order preserved)
* **Robust retries** on rate limits / timeouts / 5xx using **tenacity**
* **Graceful per-item exception handling**

> **Topic used in this demo:** Travel Itinerary Planning

---

## Requirements

* **Python** 3.9+
* Install dependencies:

  ```bash
  pip install openai tenacity python-dotenv
  ```

---

## Environment Variables

Put these in a `.env` file or export them in your shell:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment>      # e.g., gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-07-01-preview
```

---

## How It Works

### a. Function Calling

* We define a tool (function) named `submit_itinerary` with a strict JSON schema.
* We **force** the model to call that tool using:

  ```python
  tool_choice={"type": "function", "function": {"name": "submit_itinerary"}}
  ```

* The model responds with a **tool call** containing **JSON arguments** that match the schema (destination, days, `daily_plan`, budget, tips, etc.).
* We parse those arguments and return a Python `dict`. No regex parsing. No post-hoc JSON “fixing”.

### b. Batching with Order Preservation

* `batch_process(inputs, max_workers=2, sleep_between=0.2)`:

  * Runs multiple requests concurrently (default 2 threads).
  * Preserves **input order** in the output list.
  * Uses a small throttle (`sleep_between`) between completed tasks to avoid bursts.

### c. Retries & Backoff (tenacity)

* The low-level API call is wrapped with:

  * `retry_if_exception_type` on transient exceptions: `RateLimitError`, `APIError`, `APIConnectionError`, `APITimeoutError`
  * `wait_random_exponential(min=1, max=10)`
  * `stop_after_attempt(5)`
* If all retries fail, the result for that item is an error dict (the batch continues).

### d. Exceptions & Logging

* All exceptions are logged.
* Per-item failures are returned as:

  ```json
  {"error": "Transient error after retries: RateLimitError", "destination": "X"}
  ```

  or

  ```json
  {"error": "Failed to parse tool arguments: ...", "destination": "X"}
  ```

* The **batch never crashes** due to one bad item.

---

## Run the Demo

```bash
python function_calling.py
```

```txt
Result for Paris (3 days):
--------------------------
{
  "destination": "Paris",
  "days": 3,
  "highlights": [
    "Eiffel Tower",
    "Louvre Museum",
    "Montmartre"
  ],
  "budget_total": 450,
  "day1": {
    "day": 1,
    "morning": "Visit the Eiffel Tower and enjoy the views from the top.",
    "afternoon": "Explore the Louvre Museum and see the Mona Lisa and other masterpieces.",
    "evening": "Stroll through the Tuileries Garden and enjoy dinner in a nearby café.",
    "notes": "Book Eiffel Tower tickets in advance to avoid long queues."
  }
}

Result for Tokyo (5 days):
--------------------------
{
  "destination": "Tokyo",
  "days": 5,
  "highlights": [
    "Sushi at Tsukiji",
    "Ramen tasting in Shinjuku",
    "Street food in Harajuku"
  ],
  "budget_total": 750,
  "day1": {
    "day": 1,
    "morning": "Visit Tsukiji Outer Market for fresh sushi breakfast.",
    "afternoon": "Explore Ginza for upscale shopping and enjoy lunch at a high-end sushi restaurant.",
    "evening": "Visit a local izakaya in Shinjuku for dinner and drinks."
  }
}

Result for New York (4 days):
-----------------------------
{
  "destination": "New York",
  "days": 4,
  "highlights": [
    "Statue of Liberty",
    "Metropolitan Museum of Art",
    "Central Park"
  ],
  "budget_total": 600,
  "day1": {
    "day": 1,
    "morning": "Visit the Statue of Liberty and Ellis Island; take the ferry from Battery Park.",
    "afternoon": "Explore the 9/11 Memorial & Museum; spend around 2-3 hours.",
    "evening": "Stroll through Wall Street and then dinner in the Financial District."
  }
}
```

> Actual content will vary by model, but the structure is consistent due to function calling.

---
