# 🎵 Audio Codecs

A collection of audio codecs with a **standardized API** for easy integration and benchmarking.

---------------------------------------------------------------------------------------------------------

## 🚀 Available Codecs

- [BigCodec](https://arxiv.org/abs/2409.05377)
- [DAC](https://arxiv.org/abs/2306.06546)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [EnCodec](https://arxiv.org/abs/2210.13438) + [Vocos](https://arxiv.org/abs/2306.00814)
- [FocalCodec](https://arxiv.org/abs/2502.04465)
- [Mimi](https://kyutai.org/Moshi.pdf)
- [SemantiCodec](https://arxiv.org/abs/2405.00233)
- [SpeechTokenizer](https://arxiv.org/abs/2308.16692)
- [Stable Codec](https://arxiv.org/abs/2411.19842)
- [WavLM k-means](https://arxiv.org/abs/2312.09747)
- [WavTokenizer](https://arxiv.org/abs/2408.16532)

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.8 or later](https://www.python.org).
Then, clone or download and extract the repository and navigate to `<path-to-repository>`.

To install the package locally in editable mode, **without codecs (which must be installed separately)**,
open a terminal and run:

```bash
pip install -e .
```

To install **all available codecs**, run:

```bash
pip install -e .[all]
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

To check the reconstruction quality of a codec, navigate to the `<path-to-repository>/audiocodecs` directory and run:

```bash
python <codec-name>.py
```

This will generate a file named `reconstruction.wav` corresponding to `example.wav` in the same directory.

To use one of the available codecs in your script (for example EnCodec):

```python
import torchaudio
from audiocodecs import Encodec

audio_file = "audiocodecs/example.wav"
sig, sample_rate = torchaudio.load(audio_file)
model = Encodec(sample_rate=sample_rate, orig_sample_rate=24000, num_codebooks=8)
model.requires_grad_(False).eval()
toks = model.sig_to_toks(sig)
rec_sig = model.toks_to_sig(toks)
torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
```

---------------------------------------------------------------------------------------------------------

## 📈️ Downstream Tasks

Reference implementations of downstream tasks using these audio codecs can be found in the `downstream` directory.

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
