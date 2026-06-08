import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_terminal_log(name: str, log_dir: str | Path = "logs") -> Path:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(log_dir) / f"{name}_{timestamp}.log"
    log_file = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, log_file)  # type: ignore[assignment]
    print(f"terminal log: {path}")
    return path
