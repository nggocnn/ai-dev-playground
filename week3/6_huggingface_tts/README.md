# Batch Text-to-Speech with Hugging Face (MMS VITS)

Convert **multiple .txt files** into **.wav audio** using Hugging Face **MMS TTS** models via `transformers`.
Default device is **CPU**, language can be **auto-detected** (vi/en) or forced with a flag.

---

## Features

* **Batch mode**: reads all `inputs/*.txt`
* **Multi-language**: Vietnamese (`vi`) and English (`en`) out of the box
* **Auto language detection** (simple Vietnamese diacritic heuristic)
* **Drop-in models**: MMS-VITS: `facebook/mms-tts-vie` and `facebook/mms-tts-eng`
* **Standard WAV output**: 16-bit PCM, written to `outputs/<lang>/...`
* **Chunking**: long texts are split into \~250-character segments for robust synthesis
* **CPU by default**; can run on CUDA if you pass `--device cuda`

---

## Requirements

* Python 3.9+
* `pip` and a working C toolchain (standard Python environment)
* Internet access the first time to download models from the Hub

Install minimal dependencies:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```txt
transformers==4.43.2
torch>=2.2.0
```

---

## Quick Start

### Run on CPU (auto language detection)

```bash
python tts_batch_infer.py --lang auto --device cpu
```

### Find outputs

```txt
outputs/
├─ vi/vi_story_1.wav
├─ vi/vi_story_2.wav
├─ en/en_story_1.wav
└─ en/en_story_2.wav
```

---

## Usage

```bash
python tts_batch_infer.py [--inputs-dir DIR] [--out-dir DIR] [--lang {auto,vi,en}] [--device {cpu,cuda}] [--model-id HF_MODEL]
```

### Common examples

```bash
# CPU, auto-detect language per file
python tts_batch_infer.py --lang auto --device cpu

# Force Vietnamese model
python tts_batch_infer.py --lang vi --device cpu

# Force English model
python tts_batch_infer.py --lang en --device cpu

# Override model for all files (must be a compatible Hugging Face TTS model)
python tts_batch_infer.py --model-id facebook/mms-tts-eng --lang en --device cpu
```

### Arguments

| Flag           | Description                                     | Default   |
| -------------- | ----------------------------------------------- | --------- |
| `--inputs-dir` | Where to read `.txt` files                      | `inputs`  |
| `--out-dir`    | Where to write `.wav` files                     | `outputs` |
| `--lang`       | Language hint: `auto`, `vi`, `en`               | `auto`    |
| `--device`     | Compute device                                  | `cpu`     |
| `--model-id`   | Override model for **all** files (advanced use) | *(none)*  |

> Note: When `--model-id` is not set, the script uses MMS defaults:
> `vi → facebook/mms-tts-vie` and `en → facebook/mms-tts-eng`.

---

## How It Works

1. **Input reading**: loads all `.txt` from `inputs/`.
2. **Language selection**:

   * `--lang auto` → a simple diacritic check labels text as `vi` if Vietnamese diacritics appear, otherwise `en`.
   * You can force with `--lang vi` or `--lang en`.
3. **Model loading**:

   * Vietnamese → `facebook/mms-tts-vie`
   * English → `facebook/mms-tts-eng`
   * Or pass `--model-id ...` to use a different compatible model for all files.
4. **Chunking**: the script splits long text into \~250-char chunks and synthesizes each chunk, then concatenates.
5. **Saving**: waveform is clamped to \[-1, 1], converted to **16-bit PCM WAV**, and written to `outputs/<lang>/<stem>.wav`.
