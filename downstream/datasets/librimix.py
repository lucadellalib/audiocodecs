# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""LibriMix dataset."""

import csv
import logging
import os
from typing import Optional, Sequence

import speechbrain as sb
import torch
import torchaudio


__all__ = ["dataio_prepare", "prepare_data"]


# Workaround to use fastest backend (SoundFile)
try:
    torchaudio._backend.utils.get_available_backends().pop("ffmpeg", None)
except Exception:
    pass

# Logging configuration
logging.basicConfig(
    level=logging.INFO,  # format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


def prepare_data(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = ("train-100", "train-360", "dev", "test"),
    num_speakers: "int" = 2,
    add_noise: "bool" = False,
    version: "str" = "wav16k/min",
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the LibriMix dataset
    (see https://github.com/JorisCos/LibriMix).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    splits:
        The dataset splits to prepare.
    num_speakers:
        The number of speakers (1, 2 or 3).
    add_noise:
        True to add WHAM! noise, False otherwise.
    version:
        The dataset version.

    Raises
    ------
    ValueError
        If an invalid argument value is given.
    RuntimeError
        If one of the expected split folders is missing.

    Examples
    --------
    >>> # Expected folder structure: LibriMix/Libri2Mix/wav16k/min/{train-100, train-360, dev, test}/{mix_both, mix_clean, noise, s1, s2}
    >>> prepare_data("LibriMix", num_speakers=2, version="wav16k/min")

    """
    if not save_folder:
        save_folder = data_folder
    if num_speakers not in [1, 2, 3]:
        raise ValueError(f"`num_speakers` ({num_speakers}) must be either 1, 2 or 3")
    os.makedirs(save_folder, exist_ok=True)
    version = version.replace("/", os.sep)

    # Write output CSV for each split
    for split in splits:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        split_folder = os.path.join(f"Libri{num_speakers}Mix", version, split)
        if not os.path.exists(os.path.join(data_folder, split_folder)):
            raise RuntimeError(
                f"{os.path.join(data_folder, split_folder)} does not exist"
            )

        # Mix clean files
        mix_clean_folder = os.path.join(split_folder, "mix_clean")
        mix_clean_wavs = sorted(os.listdir(os.path.join(data_folder, mix_clean_folder)))
        mix_clean_wavs = [
            os.path.join("$DATA_ROOT", mix_clean_folder, mix_clean_wav)
            for mix_clean_wav in mix_clean_wavs
            if mix_clean_wav.endswith(".wav")
        ]

        # Original files
        all_src_wavs = []
        for i in range(1, num_speakers + 1):
            src_folder = os.path.join(split_folder, f"s{i}")
            src_wavs = [
                mix_clean_wav.replace(mix_clean_folder, src_folder)
                for mix_clean_wav in mix_clean_wavs
            ]
            all_src_wavs.append(src_wavs)

        if add_noise:
            # Mix both files
            mix_both_folder = os.path.join(split_folder, "mix_both")
            mix_both_wavs = [
                mix_clean_wav.replace(mix_clean_folder, mix_both_folder)
                for mix_clean_wav in mix_clean_wavs
            ]

            # Noise files
            noise_folder = os.path.join(split_folder, "noise")
            noise_wavs = [
                mix_clean_wav.replace(mix_clean_folder, noise_folder)
                for mix_clean_wav in mix_clean_wavs
            ]

        headers = (
            ["ID", "duration", "mix_clean_wav"]
            + (["mix_both_wav"] if add_noise else [])
            + [f"src{i}_wav" for i in range(1, num_speakers + 1)]
            + (["noise_wav"] if add_noise else [])
        )
        output_csv = os.path.join(save_folder, f"{split}.csv")
        _LOGGER.info(f"Writing {output_csv}...")
        with open(output_csv, "w", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            for i in range(len(mix_clean_wavs)):
                ID = f"{split}_{num_speakers}mix_{str(i).zfill(7)}"
                mix_clean_wav = mix_clean_wavs[i]
                src_wavs = [src_wavs[i] for src_wavs in all_src_wavs]
                info = sb.dataio.dataio.read_audio_info(
                    mix_clean_wav.replace("$DATA_ROOT", data_folder)
                )
                duration = info.num_frames / info.sample_rate
                entry = dict(
                    zip(
                        headers,
                        [ID, duration, mix_clean_wav]
                        + ([mix_both_wavs[i]] if add_noise else [])
                        + src_wavs
                        + ([noise_wavs[i]] if add_noise else []),
                    )
                )
                csv_writer.writerow(entry)

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info("Done!")


def dataio_prepare(
    data_folder,
    train_csv=None,
    valid_csv=None,
    test_csv=None,
    sample_rate=16000,
    train_remove_if_longer=60.0,
    valid_remove_if_longer=60.0,
    test_remove_if_longer=60.0,
    train_remove_if_shorter=0.0,
    valid_remove_if_shorter=0.0,
    test_remove_if_shorter=0.0,
    sorting="ascending",
    debug=False,
    add_noise=False,
    num_speakers=2,
    **kwargs,
):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    """
    datasets = []

    if train_csv is not None:
        train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=train_csv,
            replacements={"DATA_ROOT": data_folder},
        )
        # Sort training data to speed up training
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=sorting == "descending",
            key_max_value={"duration": train_remove_if_longer},
            key_min_value={"duration": train_remove_if_shorter},
        )
        datasets.append(train_data)
    else:
        datasets.append(None)

    if valid_csv is not None:
        valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=valid_csv,
            replacements={"DATA_ROOT": data_folder},
        )
        # Sort validation data to speed up validation
        valid_data = valid_data.filtered_sorted(
            sort_key="duration",
            reverse=not debug,
            key_max_value={"duration": valid_remove_if_longer},
            key_min_value={"duration": valid_remove_if_shorter},
        )
        datasets.append(valid_data)
    else:
        datasets.append(None)

    if test_csv is not None:
        test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=test_csv,
            replacements={"DATA_ROOT": data_folder},
        )
        # Sort the test data to speed up testing
        test_data = test_data.filtered_sorted(
            sort_key="duration",
            reverse=not debug,
            key_max_value={"duration": test_remove_if_longer},
            key_min_value={"duration": test_remove_if_shorter},
        )
        datasets.append(test_data)
    else:
        datasets.append(None)

    # Define audio pipeline
    takes = ["mix_both_wav" if add_noise else "mix_clean_wav"] + [
        f"src{i}_wav" for i in range(1, num_speakers + 1)
    ]
    provides = ["in_sig", "out_sig"]

    def audio_pipeline(mix_wav, *src_wavs):
        # Mixed signal
        original_sample_rate = sb.dataio.dataio.read_audio_info(mix_wav).sample_rate
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)

        # Source signals
        src_sigs = []
        for src_wav in src_wavs:
            assert (
                original_sample_rate
                == sb.dataio.dataio.read_audio_info(src_wav).sample_rate
            )
            src_sig = sb.dataio.dataio.read_audio(src_wav)
            src_sigs.append(src_sig)
        src_sigs = torch.stack(src_sigs)  # [S, T]

        in_sig = torchaudio.functional.resample(
            mix_sig,
            original_sample_rate,
            sample_rate,
        )
        yield in_sig

        out_sig = torchaudio.functional.resample(
            src_sigs,
            original_sample_rate,
            sample_rate,
        )
        # Flatten as SpeechBrain's dataloader does not support multichannel audio
        out_sig = out_sig.flatten()  # [ST]
        yield out_sig

    sb.dataio.dataset.add_dynamic_item(
        [x for x in datasets if x is not None], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(
        [x for x in datasets if x is not None], ["id"] + provides
    )

    return datasets
