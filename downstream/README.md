# 🔊 Downstream Tasks

A [SpeechBrain](https://speechbrain.github.io) codebase for benchmarking audio codecs on downstream tasks.

---------------------------------------------------------------------------------------------------------

## 🚀 Tasks

- Automatic speech recognition (ASR)
- Intent classification (IC)
- Keyword spotting (KS)
- Speech enhancement (SE)
- Speech emotion recognition (SER)
- Speaker identification (SI)
- Speech language modeling (SLM)
- Speech resynthesis (SR)
- Speech separation (SS)
- Text-to-speech (TTS)
- Voice conversion (VC)

---------------------------------------------------------------------------------------------------------

## 📰 Changelog

### v0.0.2 (2026-04-15)

- Added new downstream tasks, datasets, and metrics
- Introduced a more scalable experiment configuration system
- Improved the overall code structure

### v0.0.1 (2025-02-12)

- Initial release (available at https://github.com/lucadellalib/audiocodecs/tree/v0.0.1)

---------------------------------------------------------------------------------------------------------

## 📂 Datasets

### 📌 IEMOCAP

Download the dataset from the [official website](https://sail.usc.edu/iemocap/).

Extract it in a folder named `IEMOCAP`.

Expected folder structure: `IEMOCAP/{Session1, Session2, Session3, Session4, Session5}`

---------------------------------------------------------------------------------------------------------

### 📌 LibriSpeech 460

Download the following files from the [official website](https://www.openslr.org/12):

- `train-clean-100.tar.gz`
- `train-clean-360.tar.gz`
- `dev-clean.tar.gz`
- `test-clean.tar.gz`

Extract them to a folder named `LibriSpeech`.

Expected folder structure: `LibriSpeech/{train-clean-100, train-clean-360, dev-clean, test-clean}`

---------------------------------------------------------------------------------------------------------

### 📌 Libri1Mix Test

Follow the instructions from the [official repository](https://github.com/JorisCos/LibriMix).

Expected folder structure: `LibriMix/Libri1Mix/wav16k/min/test/{mix_both, mix_clean, s1}`

---------------------------------------------------------------------------------------------------------

### 📌 Libri2Mix 100

Follow the instructions from the [official repository](https://github.com/JorisCos/LibriMix).

Expected folder structure: `LibriMix/Libri2Mix/wav16k/min/{train-100, dev, test}/{mix_both, mix_clean, s1, s2}`

---------------------------------------------------------------------------------------------------------

### 📌 Mini Multilingual LibriSpeech (MiniMLS) Test

Download the dataset from [Zenodo](https://zenodo.org/records/14791114).

Extract it to a folder named `MiniMLS`.

Expected folder structure: `MiniMLS/{mls_dutch, mls_french, mls_german, mls_italian, mls_polish, mls_portuguese, mls_spanish}/test`

---------------------------------------------------------------------------------------------------------

### 📌 SLURP

Download the dataset from [Zenodo](https://zenodo.org/records/4274930).

Extract `slurp_real.tar.gz` and `slurp_synth.tar.gz` to a folder named `SLURP`.

Download all the JSONL files from the [official repository](https://github.com/pswietojanski/slurp/tree/master/dataset/slurp) into `SLURP`.

Expected folder structure: `SLURP/{slurp_real, slurp_synth, devel.jsonl, test.jsonl, train.jsonl, train_synthetic.jsonl}`

---------------------------------------------------------------------------------------------------------

### 📌 Speech Commands

Download the dataset from the [official link](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

Extract it to a folder named `SpeechCommands`.

Expected folder structure: `SpeechCommands/{_background_noise_, backward, bed, bird, cat, ..., validation_list.txt}`

---------------------------------------------------------------------------------------------------------

### 📌 VCTK

Download the `VCTK-Corpus-0.92.zip` from the [official website](https://datashare.ed.ac.uk/handle/10283/3443):

Extract them to a folder named `VCTK`.

Expected folder structure: `VCTK/{wav48_silence_trimmed}`

---------------------------------------------------------------------------------------------------------

### 📌 VoiceBank

Download the following files from the [official website](https://datashare.ed.ac.uk/handle/10283/2791):

- `clean_testset_wav.zip`
- `clean_trainset_28spk_wav.zip`
- `noisy_testset_wav.zip`
- `noisy_trainset_28spk_wav.zip`

Extract them to a folder named `VoiceBank`.

Expected folder structure: `VoiceBank/{clean_testset_wav, clean_trainset_28spk_wav, noisy_testset_wav, noisy_trainset_28spk_wav}`

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.8 or later](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>/downstream`.

### Using Conda (recommended)

Make sure that `conda` is installed (see the [installation guide](https://docs.anaconda.com/miniconda/install/)).
Open a terminal and run:

```bash
conda env create -n audiocodecs-env -f environment.yml
```

To activate the virtual environment:

```bash
conda activate audiocodecs-env
```

### Using Pip

Open a terminal and run:

```bash
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

If the cluster compute nodes do not have internet access, you can download the model weights on the login nodes.
To do so, open a terminal and run:

```bash
python download.py
```

To make the experiment configuration system more scalable, we introduce a new feature: **YAML file merging**.

Instead of having a single YAML file defining everything, from the codec to the dataset and downstream architecture,
you can now use separate configuration files for each component.

The main script will automatically merge them into a single `config.yaml` file, which is saved in the experiment results directory.

The new configuration system is backward compatible, meaning that you can still define everything in a single YAML file if you prefer (see `hparams/_legacy`).
As a result, the merged `config.yaml` can also be run directly later to reproduce an experiment.

The only requirement is that there must be no duplicate keys across the YAML files being combined. See the examples in `hparams`.

To run an experiment, navigate to `<path-to-repository>/downstream`, open a terminal and run:

```bash
python {train,test}_<task>.py \
hparams/<task>/<config>.yaml hparams/<codec>/<config>.yaml hparams/<dataset>/<config>.yaml \
--data_folder <path-to-data-folder>
```

This command automatically creates a `results` directory, where all logs, checkpoints, metrics, etc. will be stored.

For example:

```bash
python test_sr.py \
hparams/tasks/sr.yaml hparams/codecs/bigcodec.yaml hparams/datasets/librispeech-test.yaml \
--data_folder data/LibriSpeech \
--save_audios True
```

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