# Delay Reason Classification — LLM-Only (Azure OpenAI)

Classify free-text delivery/maintenance logs into operational delay reasons **using Azure OpenAI**. Results are written to a CSV.

---

## What this does

* **Input:** a list of free-text log entries (the script includes 10 sample logs).
* **Output:** `maintenance_logs.csv` with columns:

```txt
log_id, log_entry, label
```

* **Label set:**

  * Traffic
  * Customer Issue
  * Vehicle Issue
  * Weather
  * Sorting/Labeling Error
  * Human Error
  * Technical System Failure
  * Other

The prompt constrains the model to return **exactly one** of the above categories (temperature=0 for determinism). A small sanitizer maps near-variants back to the canonical labels; unknowns default to **Other**.

---

## Requirements

* **Python** 3.9+
* Install dependencies:

  ```bash
  pip install openai python-dotenv
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

## Run

```bash
python maintenance_logs.py
```

Change output path with an env var:

```bash
OUTPUT_CSV=outputs/results.csv python maintenance_logs.py
```

---

## Output (sample)

```csv
log_id,log_entry,label
1,Driver reported heavy traffic on highway due to construction,Traffic
2,"Package not accepted, customer unavailable at given time",Customer Issue
3,"Vehicle engine failed during route, replacement dispatched",Vehicle Issue
4,Unexpected rainstorm delayed loading at warehouse,Weather
5,"Sorting label missing, required manual barcode scan",Sorting/Labeling Error
6,Driver took a wrong turn and had to reroute,Human Error
7,"No issue reported, arrived on time",Other
8,"Address was incorrect, customer unreachable",Customer Issue
9,System glitch during check-in at loading dock,Technical System Failure
10,Road accident caused a long halt near delivery point,Traffic
```

---

## How it works

1. **Single LLM call per log**
   The prompt lists the allowed categories and brief decision rules. The model returns one label.

2. **Sanitization**
   Output is normalized and mapped to the canonical set (e.g., “technical failure” → “Technical System Failure”). Anything unexpected resolves to **Other** to keep the CSV clean.

3. **Determinism & cost**
   `temperature=0`. You can batch large volumes by adding concurrency controls and retries if needed.

---
