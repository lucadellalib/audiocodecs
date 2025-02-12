# 🔊 Downstream Tasks

A [SpeechBrain](https://speechbrain.github.io) codebase for benchmarking audio codecs on downstream tasks.

---------------------------------------------------------------------------------------------------------

## 🚀 Implemented Tasks

- Automatic speech recognition (ASR)
- Speech enhancement (SE)
- Speech emotion recognition (SER)
- Speaker identification (SI)
- Speech resynthesis (SR)
- Speech separation (SS)
- Text-to-speech (TTS)
- Voice conversion (VC)

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

To run an experiment, navigate to `<path-to-repository>/downstream`, open a terminal and run:

```bash
python {train,test}_<task>.py hparams/<task>/<dataset>/<config>.yaml --data_folder <path-to-data-folder>
```

This command automatically creates a `results` directory, where all logs, checkpoints, metrics, etc. will be stored.

---------------------------------------------------------------------------------------------------------

## @ Citing

```
@article{dellalibera2025focalcodec,
    title   = {{FocalCodec}: Low-Bitrate Speech Coding via Focal Modulation Networks},
    author  = {Luca {Della Libera} and Francesco Paissan and Cem Subakan and Mirco Ravanelli},
    journal = {arXiv preprint arXiv:2502.04465},
    year    = {2025},
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