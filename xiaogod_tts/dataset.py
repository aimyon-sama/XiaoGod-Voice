import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .audio import load_wav, wav_to_log_mel
from .config import AudioConfig
from .text import Vocabulary


@dataclass
class TTSItem:
    wav_path: str
    text: str


def read_metadata(path: str | Path) -> list[TTSItem]:
    items: list[TTSItem] = []
    base = Path(path).parent
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 2:
                continue
            wav_path = row[0].strip()
            if not Path(wav_path).is_absolute():
                cwd_path = Path(wav_path)
                wav_path = str(cwd_path.resolve() if cwd_path.exists() else (base / wav_path).resolve())
            items.append(TTSItem(wav_path=wav_path, text=row[1].strip()))
    if not items:
        raise ValueError(f"No valid rows found in metadata: {path}")
    return items


def make_uniform_durations(token_count: int, mel_len: int) -> torch.Tensor:
    token_count = max(token_count, 1)
    base = mel_len // token_count
    rem = mel_len % token_count
    durations = torch.full((token_count,), max(base, 1), dtype=torch.long)
    if rem > 0:
        durations[:rem] += 1
    diff = int(durations.sum().item() - mel_len)
    if diff > 0:
        for i in range(diff):
            j = token_count - 1 - (i % token_count)
            durations[j] = max(durations[j] - 1, 1)
    return durations


class TTSDataset(Dataset):
    def __init__(self, items: list[TTSItem], vocab: Vocabulary, audio_cfg: AudioConfig):
        self.items = items
        self.vocab = vocab
        self.audio_cfg = audio_cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        tokens = torch.tensor(self.vocab.encode(item.text), dtype=torch.long)
        wav = load_wav(item.wav_path, self.audio_cfg.sample_rate)
        mel = wav_to_log_mel(wav, self.audio_cfg)
        durations = make_uniform_durations(tokens.numel(), mel.size(0))
        return {
            "tokens": tokens,
            "mel": mel,
            "durations": durations,
            "text": item.text,
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    tokens = pad_sequence([b["tokens"] for b in batch], batch_first=True, padding_value=pad_id)
    durations = pad_sequence([b["durations"] for b in batch], batch_first=True, padding_value=0)
    mels = pad_sequence([b["mel"] for b in batch], batch_first=True, padding_value=0.0)
    token_lens = torch.tensor([b["tokens"].numel() for b in batch], dtype=torch.long)
    mel_lens = torch.tensor([b["mel"].size(0) for b in batch], dtype=torch.long)
    return {
        "tokens": tokens,
        "durations": durations,
        "mels": mels,
        "token_lens": token_lens,
        "mel_lens": mel_lens,
    }
