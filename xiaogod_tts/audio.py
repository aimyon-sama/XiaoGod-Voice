import torch
import torchaudio
from torchaudio.transforms import GriffinLim, InverseMelScale, MelSpectrogram

from .config import AudioConfig


def load_wav(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)
    peak = wav.abs().max().clamp_min(1e-5)
    return wav / peak * 0.95


def mel_transform(cfg: AudioConfig) -> MelSpectrogram:
    return MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        n_mels=cfg.n_mels,
        power=1.0,
        normalized=False,
    )


def wav_to_log_mel(wav: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    mel = mel_transform(cfg)(wav)
    return torch.log(torch.clamp(mel, min=1e-5)).transpose(0, 1)


def log_mel_to_wav(log_mel: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    mel = torch.exp(log_mel.transpose(0, 1).cpu()).clamp_min(1e-5)
    inv_mel = InverseMelScale(
        n_stft=cfg.n_fft // 2 + 1,
        n_mels=cfg.n_mels,
        sample_rate=cfg.sample_rate,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    spec = inv_mel(mel).clamp_min(1e-5)
    gl = GriffinLim(
        n_fft=cfg.n_fft,
        n_iter=cfg.griffin_lim_iters,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        power=1.0,
    )
    wav = gl(spec)
    return wav / wav.abs().max().clamp_min(1e-5) * 0.95
