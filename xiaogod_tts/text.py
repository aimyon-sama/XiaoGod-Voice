import json
import re
from pathlib import Path

try:
    from pypinyin import Style, pinyin
except ImportError:  # pragma: no cover
    pinyin = None
    Style = None


PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

_punct_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _punct_re.sub("", text)
    return text


def text_to_units(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if pinyin is None:
        return list(text)
    units: list[str] = []
    for item in pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors=lambda x: list(x)):
        token = item[0]
        if token.strip():
            units.append(token)
    return units


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = token_to_id or {PAD: 0, UNK: 1, BOS: 2, EOS: 3}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def build(self, texts: list[str]) -> None:
        for text in texts:
            for unit in text_to_units(text):
                self.add(unit)

    def encode(self, text: str) -> list[int]:
        ids = [self.token_to_id[BOS]]
        ids.extend(self.token_to_id.get(unit, self.token_to_id[UNK]) for unit in text_to_units(text))
        ids.append(self.token_to_id[EOS])
        return ids

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
