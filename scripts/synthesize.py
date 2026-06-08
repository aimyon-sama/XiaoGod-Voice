import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from xiaogod_tts.audio import log_mel_to_wav
from xiaogod_tts.config import AudioConfig, ModelConfig
from xiaogod_tts.logging import setup_terminal_log
from xiaogod_tts.model import XiaoGodTTS
from xiaogod_tts.text import Vocabulary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--speaker", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_terminal_log("synthesize")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    audio_cfg = AudioConfig(**ckpt["config"]["audio"])
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    vocab = Vocabulary(ckpt["vocab"])
    speakers = ckpt["speakers"]
    if args.speaker not in speakers:
        raise ValueError(f"Unknown speaker '{args.speaker}'. Available: {', '.join(speakers)}")

    model = XiaoGodTTS(model_cfg, audio_cfg.n_mels, vocab.pad_id).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokens = torch.tensor([vocab.encode(args.text)], dtype=torch.long, device=args.device)
    speaker_id = torch.tensor([speakers[args.speaker]], dtype=torch.long, device=args.device)
    with torch.no_grad():
        mel = model.infer(tokens, speaker_id, speed=args.speed)
    wav = log_mel_to_wav(mel, audio_cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav.cpu().numpy(), audio_cfg.sample_rate)
    print(json.dumps({"out": args.out, "speaker": args.speaker, "sample_rate": audio_cfg.sample_rate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
