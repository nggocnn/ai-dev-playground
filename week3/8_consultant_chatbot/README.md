# Laptop Consultant Chatbot (Azure OpenAI + ChromaDB)

## Features

* **Two modes**:

  * `embed`: build/update the vector index from `laptops.csv`
  * `query` *(default)*: read `queries.txt`, search Chroma, and ask the LLM for recommendations
* **Strict schema** for data files (no guessing): `id,name,description,tags`
* **Persistent** Chroma store (embed once, reuse many times)
* **TQDM** progress bar during embedding
* Strong, **grounded system prompt** (recommends only from retrieved context)

---

## Repo Layout (suggested)

```txt
.
├── consultant_chatbot.py
├── .env
├── laptops.csv
├── queries.txt
└── chroma/               # created at runtime (persistent DB)
```

---

## Requirements

```bash
python 3.9+
pip install -r requirements.txt
```

---

## Configure Azure OpenAI

> In **Azure**, `model=` must be your **deployment name**, not the base model ID.

Create a `.env` file:

```env
AZURE_OPENAI_API_VERSION=2024-07-01-preview

AZURE_OPENAI_EMBEDDING_ENDPOINT=https://<your-embed-resource>.openai.azure.com/
AZURE_OPENAI_EMBEDDING_API_KEY=YOUR_EMBED_KEY
AZURE_OPENAI_EMBED_MODEL=<your-embedding-deployment-name>

AZURE_OPENAI_LLM_ENDPOINT=https://<your-llm-resource>.openai.azure.com/
AZURE_OPENAI_LLM_API_KEY=YOUR_CHAT_KEY
AZURE_OPENAI_LLM_MODEL=<your-chat-deployment-name>

CHROMA_DIR=./chroma
CHROMA_COLLECTION_NAME=laptops
```

---

## Data Files

### `laptops.csv` (required for `--mode embed`)

Strict headers: `id,name,description,tags`

```csv
id,name,description,tags
1,Gaming Beast Pro,A high-end gaming laptop with RTX 4080, 32GB RAM, and 1TB SSD. Perfect for hardcore gaming.,gaming;high-performance;windows
2,Business Ultrabook X1,Lightweight business ultrabook with long battery life.,business;ultrabook;lightweight
3,Student Basic,Affordable laptop ideal for study and browsing.,student;budget;general
```

* Keep `description` concise but **specific** (CPU/GPU/RAM/SSD/display/weight/battery/OS/use case).
* Tags: semicolon-separated keywords (e.g., `gaming;windows;rtx4060`).

### `queries.txt` (used by `--mode query`)

One query per line, `#` lines are ignored.

```txt
I travel a lot for work. Under 1.3 kg, all-day battery, Windows.
Best value gaming laptop with RTX 4060.
Mac for Final Cut Pro and Lightroom — 14" preferred.
```

---

## Run It

### Embed (one-time)

```bash
python consultant_chatbot.py --mode embed --laptops ./laptops.csv --chroma_dir ./chroma --collection laptops --reset
```

* `--reset` drops and recreates the collection (useful after big data changes).

### Query (default)

```bash
python consultant_chatbot.py --queries ./queries.txt --n_results 3 --chroma_dir ./chroma --collection laptops
# or simply:
python consultant_chatbot.py
```

---

## CLI Reference

```bash
python consultant_chatbot.py [--mode embed|query]
                             [--laptops ./laptops.csv]
                             [--queries ./queries.txt]
                             [--n_results 3]
                             [--chroma_dir ./chroma]
                             [--collection laptops]
                             [--reset]
```

## How It Works

1. **Embed mode**

   * Loads `laptops.csv`
   * Builds `combined_text = "Name: ...\nDescription: ...\nTags: ..."`
   * Gets embeddings via Azure OpenAI
   * Adds `(id, embedding, document=combined_text, metadata={name,tags})` to Chroma

2. **Query mode**

   * Embeds each query string
   * Retrieves top-K docs from Chroma
   * Builds a **grounding context** string
   * Calls Azure OpenAI Chat with a strict **system prompt** to recommend *only* from context

---
