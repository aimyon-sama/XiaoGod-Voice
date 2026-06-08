from dataclasses import dataclass, asdict


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    griffin_lim_iters: int = 48


@dataclass
class ModelConfig:
    vocab_size: int
    speaker_count: int
    hidden: int = 256
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    max_duration: int = 80


@dataclass
class TrainConfig:
    batch_size: int = 12
    epochs: int = 100
    lr: float = 2e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 1234


def to_dict(audio: AudioConfig, model: ModelConfig, train: TrainConfig | None = None) -> dict:
    data = {"audio": asdict(audio), "model": asdict(model)}
    if train is not None:
        data["train"] = asdict(train)
    return data
