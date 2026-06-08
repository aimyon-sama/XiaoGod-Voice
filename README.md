# XiaoGod Voice

一个从零训练的轻量中文 TTS 项目，支持：

- 输入中文文本并合成语音
- 使用真人录音继续训练/微调以改变音色
- 多说话人训练，合成时指定一个 speaker 输出固定音色

这个实现使用 FastSpeech 风格声学模型生成 mel 频谱，再用 Griffin-Lim 声码器还原 wav。它适合学习、原型和小数据实验；如果要达到商用音质，需要更强的对齐器和神经声码器。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 数据格式

准备一个 `metadata.csv`，每行：

```text
wav_path|text|speaker
```

示例：

```text
data/wavs/0001.wav|你好，欢迎使用小神语音。|speaker_a
data/wavs/0002.wav|今天天气不错。|speaker_a
data/wavs/1001.wav|欢迎收听今天的内容。|speaker_b
```

如果只训练一个人，也可以省略第三列，脚本会使用 `default` 作为 speaker 名称。

建议录音参数：

- 单声道 wav
- 22050 Hz 或更高，脚本会自动重采样
- 每句 2 到 12 秒
- 同一个说话人至少 30 分钟，越多越好
- 文本必须和录音完全一致

## 训练

```powershell
python scripts/train.py --metadata data/metadata.csv --out runs/base
```

## 合成

```powershell
python scripts/synthesize.py --checkpoint runs/base/latest.pt --text "你好，我是小神语音。" --speaker speaker_a --out out.wav
```

## 用真人声音微调音色

先准备目标真人的录音清单，例如 `data/new_speaker.csv`。

```powershell
python scripts/finetune.py --checkpoint runs/base/latest.pt --metadata data/new_voice.csv --speaker new_voice --out runs/new_voice
```

然后合成：

```powershell
python scripts/synthesize.py --checkpoint runs/new_voice/latest.pt --text "这是一段新的音色测试。" --speaker new_voice --out new_voice.wav
```

## 注意

这个项目是真正从零训练，不包含预训练权重。没有训练好的 checkpoint 时无法直接得到自然语音。小数据微调前，最好先用较大的中文单人或多人数据集训练基础模型。多人数据训练后，合成时通过 `--speaker` 固定输出某一个说话人。
