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
from xiaogod_tts.dataset import TTSDataset, collate_batch, read_metadata
from xiaogod_tts.logging import setup_terminal_log
from xiaogod_tts.model import XiaoGodTTS, lengths_to_mask
from xiaogod_tts.text import Vocabulary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--freeze-encoder", action="store_true")
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
    setup_terminal_log("finetune")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    audio_cfg = AudioConfig(**ckpt["config"]["audio"])
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    vocab = Vocabulary(ckpt["vocab"])

    items = read_metadata(args.metadata)

    model = XiaoGodTTS(model_cfg, audio_cfg.n_mels, vocab.pad_id).to(args.device)
    model.load_state_dict(ckpt["model"])
    if args.freeze_encoder:
        for module in (model.token_emb, model.encoder):
            for p in module.parameters():
                p.requires_grad = False

    dataset = TTSDataset(items, vocab, audio_cfg)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=partial(collate_batch, pad_id=vocab.pad_id),
    )
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=train_cfg.lr)

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total = 0.0
        bar = tqdm(loader, desc=f"finetune {epoch}/{train_cfg.epochs}")
        for batch in bar:
            tokens = batch["tokens"].to(args.device)
            durations = batch["durations"].to(args.device)
            mels = batch["mels"].to(args.device)
            mel_lens = batch["mel_lens"].to(args.device)
            token_lens = batch["token_lens"].to(args.device)

            out_batch = model(tokens, durations)
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
            bar.set_postfix(loss=f"{loss.item():.4f}")

        new_ckpt = {
            "model": model.state_dict(),
            "config": to_dict(audio_cfg, model_cfg, train_cfg),
            "vocab": vocab.token_to_id,
            "epoch": epoch,
        }
        torch.save(new_ckpt, out_dir / "latest.pt")
        print(f"epoch={epoch} loss={total / max(len(loader), 1):.4f}")

    vocab.save(out_dir / "vocab.json")
    (out_dir / "config.json").write_text(json.dumps(to_dict(audio_cfg, model_cfg, train_cfg), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
