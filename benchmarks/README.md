# Audio Codecs Benchmark

A [SpeechBrain](https://speechbrain.github.io) codebase for benchmarking audio codecs on downstream tasks.

The following tasks have been implemented so far:

- Automatic speech recognition (ASR)
- Speech enhancement (SE)
- Speech emotion recognition (SER)
- Speaker identification (SI)
- Speech separation (SS)
- Text-to-speech (TTS)

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

---------------------------------------------------------------------------------------------------------

### LibriSpeech 100h

Download the following files from the [official website](https://www.openslr.org/12):

- `train-clean-100.tar.gz`
- `dev-clean.tar.gz`
- `test-clean.tar.gz`

Extract them to a folder named `LibriSpeech`.

Expected folder structure: `LibriSpeech/{train-clean-100, dev-clean, test-clean}`

---------------------------------------------------------------------------------------------------------

### VoiceBank

Download the following files from the [official website](https://datashare.ed.ac.uk/handle/10283/2791):

- `clean_testset_wav.zip`
- `clean_trainset_28spk_wav.zip`
- `noisy_testset_wav.zip`
- `noisy_trainset_28spk_wav.zip`

Extract them to a folder named `VoiceBank`.

Expected folder structure: `VoiceBank/{clean_testset_wav, clean_trainset_28spk_wav, noisy_testset_wav, noisy_trainset_28spk_wav}`

---------------------------------------------------------------------------------------------------------

### IEMOCAP

Download the dataset from the [official website](https://sail.usc.edu/iemocap/).

Extract it in a folder named `IEMOCAP`.

Expected folder structure: `IEMOCAP/{Session1, Session2, Session3, Session4, Session5}`

---------------------------------------------------------------------------------------------------------

### Libri2Mix 100h

Follow the instructions from the [official repository](https://github.com/JorisCos/LibriMix).

Expected folder structure: `LibriMix/Libri2Mix/wav16k/min/{train-100, dev, test}/{mix_both, mix_clean, noise, s1, s2}`

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.8 or later](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a terminal and run:

```bash
# Install audiocodecs package locally in editable mode
pip install -e .[all]
```

Then, navigate to `<path-to-repository>/benchmarks`, open a terminal and run:

```bash
# Install additional dependencies
pip install -r requirements.txt
```

If no connection is available on the cluster's compute nodes, open a terminal and run:

```bash
# Download model weights
python download.py
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `<path-to-repository>/benchmarks`, open a terminal and run:

```bash
python train_<task>.py hparams/<task>/<dataset>/<config>.yaml --data_folder <path-to-data-folder>
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------