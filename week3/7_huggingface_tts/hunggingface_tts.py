import argparse
import re
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from transformers import VitsModel, AutoTokenizer

# Default mapping (can be overridden with --model-id)
DEFAULT_MODELS: Dict[str, str] = {
    "vi": "facebook/mms-tts-vie",
    "en": "facebook/mms-tts-eng",
}

# Simple Vietnamese diacritic check for lightweight auto-lang detection
VI_DIACRITICS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def pick_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        explicit = explicit.lower()
        if explicit not in {"cpu", "cuda"}:
            raise ValueError("--device must be 'cpu' or 'cuda'")
        if explicit == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(explicit)
    # Default to CPU as requested
    return torch.device("cpu")


def detect_lang(text: str) -> str:
    # Heuristic: if any Vietnamese diacritic present -> vi else en
    for ch in text.lower():
        if ch in VI_DIACRITICS:
            return "vi"
    return "en"


def chunk_text(text: str, max_chars: int = 250) -> List[str]:
    """
    Split long text into shorter chunks for robustness (pauses at sentence ends).
    """
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_chars:
        return [text]
    # Split by sentence-ish delimiters
    sentences = re.split(r"(?<=[\.\?\!\…])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + 1 + len(s) <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def save_wav_int16(waveform: torch.Tensor, sr: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim == 2:
        waveform = waveform[0]
    elif waveform.ndim != 1:
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")

    w = torch.clamp(waveform, -1.0, 1.0)
    w = (w * 32767.0).round().to(torch.int16).cpu().numpy()

    with wave.open(str(out_path), "wb") as wf:
        wf.setparams((1, 2, int(sr), w.shape[0], "NONE", "not compressed"))
        wf.writeframes(w.tobytes())


def load_model_and_tokenizer(model_id: str, device: torch.device) -> Tuple[VitsModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def synthesize_text(model: VitsModel, tokenizer: AutoTokenizer, text: str, device: torch.device) -> Tuple[torch.Tensor, int]:
    chunks = chunk_text(text, max_chars=250)
    sr = getattr(model.config, "sampling_rate", 16000)
    pieces = []
    with torch.no_grad():
        for ch in chunks:
            inputs = tokenizer(ch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs).waveform  # (1, T)
            pieces.append(out.cpu())
    # Concatenate along time axis
    full = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
    return full[0], sr  # (T,), sr


def main():
    ap = argparse.ArgumentParser(description="Batch TTS with Hugging Face MMS-VITS (Transformers).")
    ap.add_argument("--inputs-dir", type=str, default="inputs", help="Directory with .txt files")
    ap.add_argument("--out-dir", type=str, default="outputs", help="Directory to write WAVs")
    ap.add_argument("--lang", type=str, default="auto", choices=["auto", "vi", "en"], help="Language hint per file")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device (default cpu)")
    ap.add_argument("--model_id", type=str, default=None, help="Override model id (applies to all files)")
    args = ap.parse_args()

    device = pick_device(args.device)
    inputs_dir = Path(args.inputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_paths = sorted(Path(inputs_dir).glob("*.txt"))
    if not txt_paths:
        print(f"[WARN] No .txt files found in {inputs_dir}/")
        return

    # Cache models/tokenizers per language id
    cache: Dict[str, Tuple[VitsModel, AutoTokenizer]] = {}

    for p in txt_paths:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            print(f"[SKIP] {p.name}: empty content")
            continue

        if args.lang == "auto":
            lang = detect_lang(raw)
        else:
            lang = args.lang

        # Choose model
        if args.model_id:
            model_id = args.model_id
            model_key = f"override::{model_id}"
            subdir = lang  # keep language subdir for organization
        else:
            model_id = DEFAULT_MODELS[lang]
            model_key = model_id
            subdir = lang

        # Load or reuse
        if model_key not in cache:
            print(f"[INFO] Loading {model_id} on {device} for lang={lang}")
            cache[model_key] = load_model_and_tokenizer(model_id, device)
        model, tok = cache[model_key]

        print(f"[RUN ] {p.name} -> {model_id} ({lang})")
        wav, sr = synthesize_text(model, tok, raw, device)

        # outputs/<lang>/<stem>.wav
        out_path = out_dir / subdir / (p.stem + ".wav")
        save_wav_int16(wav, sr, out_path)
        print(f"[OK  ] Wrote {out_path} (sr={sr}, samples={wav.numel()})")

    print("[DONE] All files processed.")


if __name__ == "__main__":
    main()
