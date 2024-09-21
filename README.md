# Audio Codecs

A collection of audio codecs with a standardized API. The following codecs are currently supported:

- [DAC](https://arxiv.org/abs/2306.06546)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [EnCodec](https://arxiv.org/abs/2210.13438) + [Vocos](https://arxiv.org/abs/2306.00814)
- [Mimi](https://kyutai.org/Moshi.pdf)
- [SemantiCodec](https://arxiv.org/abs/2405.00233)
- [SpeechTokenizer](https://arxiv.org/abs/2308.16692)
- [WavLM k-means](https://arxiv.org/abs/2312.09747)
- [WavTokenizer](https://arxiv.org/abs/2408.16532)

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

### From source

First of all, install [Python 3.8 or later](https://www.python.org). Open a terminal and run:

```bash
pip install git+https://github.com/lucadellalib/audio-codecs.git@main
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

To use one of the available codecs in your script:

```python
import torch
import torchaudio
from audio_codecs import Encodec

sig, sample_rate = torchaudio.load("<path-to-audio-file>")
model = Encodec(sample_rate=sample_rate, orig_sample_rate=24000, num_codebooks=8)
with torch.no_grad():
    toks = model.sig_to_toks(sig)
    rec_sig = model.toks_to_sig(toks)
torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
