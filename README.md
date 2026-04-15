# 🎵 Audio Codecs

![License](https://img.shields.io/github/license/lucadellalib/audiocodecs)
![Stars](https://img.shields.io/github/stars/lucadellalib/audiocodecs?style=social)

A collection of audio codecs with a **standardized API** for easy integration and benchmarking.

---------------------------------------------------------------------------------------------------------

## 🚀 Available Codecs

- [AudioDec](https://arxiv.org/abs/2305.16608)
- [BiCodec](https://arxiv.org/abs/2503.01710)
- [BigCodec](https://arxiv.org/abs/2409.05377)
- [DAC](https://arxiv.org/abs/2306.06546)
- [DyCAST](https://arxiv.org/abs/2601.23174)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [EnCodec](https://arxiv.org/abs/2210.13438) + [Vocos](https://arxiv.org/abs/2306.00814)
- [FocalCodec](https://arxiv.org/abs/2502.04465)
- [FocalCodec-Stream](https://arxiv.org/abs/2509.16195)
- [HILCodec](https://arxiv.org/abs/2405.04752)
- [MagiCodec](https://arxiv.org/abs/2506.00385)
- [Mimi](https://kyutai.org/Moshi.pdf)
- [NanoCodec](https://arxiv.org/abs/2508.05835)
- [PAST](https://arxiv.org/abs/2505.14470)
- [SemantiCodec](https://arxiv.org/abs/2405.00233)
- [SpeechTokenizer](https://arxiv.org/abs/2308.16692)
- [Stable Codec](https://arxiv.org/abs/2411.19842)
- [WavLM + K-means](https://arxiv.org/abs/2312.09747)
- [WavTokenizer](https://arxiv.org/abs/2408.16532)
- [X-Codec 2.0](https://arxiv.org/abs/2502.04128)

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.8 or later](https://www.python.org).

To install the package **with all available codecs**, open a terminal and run:

```bash
pip install audiocodecs@git+https://github.com/lucadellalib/audiocodecs.git@main#egg=audiocodecs[all]
```

If you encounter issues (e.g. codec installation conflicts with certain PyTorch versions or platforms),
you can install the package **without codecs**, and install the codec manually as needed:

```bash
pip install audiocodecs@git+https://github.com/lucadellalib/audiocodecs.git@main#egg=audiocodecs
```

**NOTE**: the initial **v0.0.1** release is still available at https://github.com/lucadellalib/audiocodecs/tree/v0.0.1.
It can be installed with `pip install audiocodecs@git+https://github.com/lucadellalib/audiocodecs.git@v0.0.1#egg=audiocodecs`.

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

## @ Citing

```
@inproceedings{dellalibera2025focalcodec,
    title     = {{FocalCodec}: Low-Bitrate Speech Coding via Focal Modulation Networks},
    author    = {Luca {Della Libera} and Francesco Paissan and Cem Subakan and Mirco Ravanelli},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2025},
}
```

```
@inproceedings{dellalibera2026focalcodecstream,
    title     = {{FocalCodec-Stream}: Streaming Low-Bitrate Speech Coding via Causal Distillation},
    author    = {Luca {Della Libera} and Cem Subakan and Mirco Ravanelli},
    booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year      = {2026},
}
```

```
@article{dellalibera2026dycast,
    title   = {Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization},
    author  = {Luca {Della Libera} and Cem Subakan and Mirco Ravanelli},
    journal = {arXiv preprint arXiv:2601.23174},
    year    = {2026},
}
```

```
@article{speechbrainV1,
    title   = {Open-Source Conversational {AI} with {SpeechBrain} 1.0},
    author  = {Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan
               and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca {Della Libera} and Artem Ploujnikov
               and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang
               and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun
               and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Ha Nguyen
               and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Ga{{\"e}}lle Laperri{{\`e}}re
               and Mickael Rouvier and Renato De Mori and Yannick Est{{\`e}}ve},
    journal = {Journal of Machine Learning Research (JMLR)},
    year    = {2024},
    volume  = {25},
    number  = {333},
    pages   = {1--11},
}
```

```
@article{ravanelli2021speechbrain,
    title   = {{SpeechBrain}: A General-Purpose Speech Toolkit},
    author  = {Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell
               and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong
               and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva
               and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
    journal = {arXiv preprint arXiv:2106.04624},
    year    = {2021},
}
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
