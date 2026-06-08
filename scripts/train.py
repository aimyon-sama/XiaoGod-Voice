import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from xiaogod_tts.config import AudioConfig, ModelConfig, TrainConfig, to_dict
from xiaogod_tts.dataset import TTSDataset, build_speakers, collate_batch, read_metadata
from xiaogod_tts.logging import setup_terminal_log
from xiaogod_tts.model import XiaoGodTTS, lengths_to_mask
from xiaogod_tts.text import Vocabulary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def masked_l1(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    max_len = min(pred.size(1), target.size(1))
    pred = pred[:, :max_len]
    target = target[:, :max_len]
    mask = ~lengths_to_mask(lengths.clamp(max=max_len), max_len)
    return F.l1_loss(pred[mask], target[mask])


def main() -> None:
    args = parse_args()
    setup_terminal_log("train")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    items = read_metadata(args.metadata)
    vocab = Vocabulary()
    vocab.build([item.text for item in items])
    speakers = build_speakers(items)
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig(vocab_size=len(vocab), speaker_count=len(speakers))

    vocab.save(out / "vocab.json")
    (out / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(to_dict(audio_cfg, model_cfg, train_cfg), indent=2), encoding="utf-8")

    dataset = TTSDataset(items, vocab, speakers, audio_cfg)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=partial(collate_batch, pad_id=vocab.pad_id),
    )
    model = XiaoGodTTS(model_cfg, audio_cfg.n_mels, vocab.pad_id).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch}/{train_cfg.epochs}")
        for batch in bar:
            tokens = batch["tokens"].to(args.device)
            speakers_t = batch["speakers"].to(args.device)
            durations = batch["durations"].to(args.device)
            mels = batch["mels"].to(args.device)
            mel_lens = batch["mel_lens"].to(args.device)
            token_lens = batch["token_lens"].to(args.device)

            out_batch = model(tokens, speakers_t, durations)
            mel_loss = masked_l1(out_batch["mel"], mels, mel_lens)
            log_target = torch.log(durations.float().clamp_min(1.0))
            token_mask = ~lengths_to_mask(token_lens, tokens.size(1))
            dur_loss = F.mse_loss(out_batch["log_duration"][token_mask], log_target[token_mask])
            loss = mel_loss + 0.1 * dur_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            total += float(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}", mel=f"{mel_loss.item():.4f}")

        ckpt = {
            "model": model.state_dict(),
            "config": to_dict(audio_cfg, model_cfg, train_cfg),
            "vocab": vocab.token_to_id,
            "speakers": speakers,
            "epoch": epoch,
        }
        torch.save(ckpt, out / "latest.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, out / f"epoch_{epoch:04d}.pt")
        print(f"epoch={epoch} loss={total / max(len(loader), 1):.4f}")


if __name__ == "__main__":
    main()
