import torch
from torch import nn

from .config import ModelConfig


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = int(max_len or lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) >= lengths.unsqueeze(1)


def length_regulate(encoded: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reps: list[torch.Tensor] = []
    lens: list[int] = []
    for enc, dur in zip(encoded, durations):
        dur = dur.clamp(min=0).long()
        expanded = torch.repeat_interleave(enc, dur, dim=0)
        if expanded.numel() == 0:
            expanded = enc[:1]
        reps.append(expanded)
        lens.append(expanded.size(0))
    out = nn.utils.rnn.pad_sequence(reps, batch_first=True)
    return out, torch.tensor(lens, dtype=torch.long, device=encoded.device)


class DurationPredictor(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.norm1(torch.relu(x)))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.norm2(torch.relu(x)))
        return self.proj(x).squeeze(-1)


class XiaoGodTTS(nn.Module):
    def __init__(self, cfg: ModelConfig, n_mels: int, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden, padding_idx=pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.heads,
            dim_feedforward=cfg.hidden * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.heads,
            dim_feedforward=cfg.hidden * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.layers)
        self.duration = DurationPredictor(cfg.hidden, cfg.dropout)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=cfg.layers)
        self.mel_proj = nn.Linear(cfg.hidden, n_mels)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.eq(self.pad_id)
        x = self.token_emb(tokens)
        return self.encoder(x, src_key_padding_mask=mask)

    def forward(self, tokens: torch.Tensor, durations: torch.Tensor) -> dict:
        enc = self.encode(tokens)
        log_dur = self.duration(enc)
        expanded, mel_lens = length_regulate(enc, durations)
        mel_mask = lengths_to_mask(mel_lens, expanded.size(1))
        dec = self.decoder(expanded, src_key_padding_mask=mel_mask)
        mel = self.mel_proj(dec)
        return {"mel": mel, "log_duration": log_dur, "mel_lens": mel_lens}

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
        enc = self.encode(tokens)
        log_dur = self.duration(enc)
        dur = torch.round(torch.exp(log_dur).clamp(1, self.cfg.max_duration) / max(speed, 0.2)).long()
        expanded, mel_lens = length_regulate(enc, dur)
        mel_mask = lengths_to_mask(mel_lens, expanded.size(1))
        dec = self.decoder(expanded, src_key_padding_mask=mel_mask)
        return self.mel_proj(dec)[0, : mel_lens[0]]
