# AI-Powered Meeting Summarizer (Azure OpenAI)

This tool uses **Azure OpenAI** GPT models to process meeting transcripts and generate concise, structured summaries.

---

## Features

- **Structured output**: summaries with `## Key Points`, `## Decisions`, and `## Action Items`
- **Chunking with overlap** to preserve context across splits
- **Context note** in each chunk to minimize confusion from missing segments
- **Option to disable chunking** (`--no-chunk`) for short transcripts
- **Fully configurable** via CLI flags and `.env`

---

## Requirements

- Python 3.8+
- [Azure OpenAI resource](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) with a deployed chat model (e.g., `gpt-4o-mini`)

Install dependencies:

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

## Usage

### Basic

Process all `.txt` transcripts in `./inputs` and save summaries to `./outputs`:

```bash
python meeting_summarizer.py
```

### Disable chunking

Send the entire transcript in one API request (for short files):

```bash
python meeting_summarizer.py --no-chunk
```

### Custom chunk size and overlap

```bash
python meeting_summarizer.py \
  --transcripts-dir ./transcripts \
  --output-dir ./summaries \
  --chunk-char-limit 10000 \
  --overlap-ratio 0.2
```

---

## CLI Arguments

| Argument             | Default         | Description                                               |
| -------------------- | --------------- | --------------------------------------------------------- |
| `--transcripts-dir`  | `./inputs`      | Folder containing `.txt` transcripts (recursive search)   |
| `--output-dir`       | `./outputs`     | Folder to save summaries                                  |
| `--chunk-char-limit` | `12000`         | Approximate max characters per chunk                      |
| `--overlap-ratio`    | `0.15`          | Fraction of previous chunk's lines to overlap into next   |
| `--no-chunk`         | *(off)*         | Disable chunking and send whole transcript in one request |

---

## Output

For each transcript `meeting1.txt`, a file `meeting1_summary.txt` is created in the output folder, containing:

```txt
## Key Points
- ...
- ...

## Decisions
- ...

## Action Items
- ...
```

---

## Notes

- **Chunking** is recommended for transcripts near or above your model's token limit.
- **Overlap** helps preserve continuity between chunks but increases token usage.
- **No-chunk mode** is best for short files to avoid unnecessary splits.
- If your transcripts contain sensitive data, scrub them before sending to Azure OpenAI.

---

## Example Run

```bash
python meeting_summarizer.py --transcripts-dir ./sample_meetings --output-dir ./summaries
```
