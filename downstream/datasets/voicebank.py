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

"""VoiceBank dataset."""

import csv
import logging
import os
from typing import Optional, Sequence

import speechbrain as sb
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

# fmt: off
_TRAIN_SPEAKER_IDS = [
    "p226", "p287", "p227", "p228", "p230", "p231", "p233", "p236", "p239", "p243",
    "p244", "p250", "p254", "p256", "p258", "p259", "p267", "p268", "p269", "p270",
    "p273", "p274", "p276", "p277", "p278", "p279", "p282", "p286"
]
# fmt: on


def prepare_data(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = (
        "trainset_28spk_wav",
        "validset_wav",
        "testset_wav",
    ),
    num_valid_speakers: "int" = 2,
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the VoiceBank dataset.

    The following files must be downloaded from the official website (https://datashare.ed.ac.uk/handle/10283/2791)
    and extracted to a common folder (e.g. `VoiceBank`):
    - `clean_testset_wav.zip`
    - `clean_trainset_28spk_wav.zip`
    - `noisy_testset_wav.zip`
    - `noisy_trainset_28spk_wav.zip`

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    splits:
        The dataset splits to prepare.
    num_valid_speakers:
        The number of speakers in the training set to use for validation
        (these speakers will be removed from the training set).

    Raises
    ------
    ValueError
        If an invalid argument value is given.
    RuntimeError
        If one of the expected split folders is missing.

    Examples
    --------
    >>> # Expected folder structure:
    >>> # VoiceBank/{clean_testset_wav, clean_trainset_28spk_wav, noisy_testset_wav, noisy_trainset_28spk_wav}
    >>> prepare_data("VoiceBank")

    """
    if not save_folder:
        save_folder = data_folder
    if num_valid_speakers > len(_TRAIN_SPEAKER_IDS):
        raise ValueError(
            f"`num_valid_speakers` ({num_valid_speakers}) must be <= than the total "
            f"number of speakers in the training set ({len(_TRAIN_SPEAKER_IDS)})"
        )
    os.makedirs(save_folder, exist_ok=True)

    # Write output CSV for each split
    valid_noisy_wavs = []
    valid_clean_wavs = []
    for split in splits:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        if split != "validset_wav":
            # Noisy files
            noisy_folder = f"noisy_{split}"
            if not os.path.exists(os.path.join(data_folder, noisy_folder)):
                raise RuntimeError(
                    f"{os.path.join(data_folder, noisy_folder)} does not exist"
                )

            noisy_wavs = sorted(os.listdir(os.path.join(data_folder, noisy_folder)))
            noisy_wavs = [
                os.path.join("$DATA_ROOT", noisy_folder, noisy_wav)
                for noisy_wav in noisy_wavs
                if noisy_wav.endswith(".wav")
            ]

            # Clean files
            clean_folder = f"clean_{split}"
            if not os.path.exists(os.path.join(data_folder, clean_folder)):
                raise RuntimeError(
                    f"{os.path.join(data_folder, clean_folder)} does not exist"
                )

            clean_wavs = [
                noisy_wav.replace(noisy_folder, clean_folder)
                for noisy_wav in noisy_wavs
            ]
        else:
            noisy_wavs = valid_noisy_wavs
            clean_wavs = valid_clean_wavs

        if split == "trainset_28spk_wav":
            # Remove the given number of speakers from the training set and add them to the validation set
            valid_speaker_ids = _TRAIN_SPEAKER_IDS[:num_valid_speakers]

            train_noisy_wavs = []
            for noisy_wav in noisy_wavs:
                if any(ID in noisy_wav for ID in valid_speaker_ids):
                    valid_noisy_wavs.append(noisy_wav)
                else:
                    train_noisy_wavs.append(noisy_wav)
            noisy_wavs = train_noisy_wavs

            train_clean_wavs = []
            for clean_wav in clean_wavs:
                if any(ID in clean_wav for ID in valid_speaker_ids):
                    valid_clean_wavs.append(clean_wav)
                else:
                    train_clean_wavs.append(clean_wav)
            clean_wavs = train_clean_wavs

        assert len(noisy_wavs) == len(clean_wavs)

        headers = ["ID", "duration", "noisy_wav", "clean_wav"]
        output_csv = os.path.join(save_folder, f"{split}.csv")
        _LOGGER.info(f"Writing {output_csv}...")
        with open(output_csv, "w", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            for i in range(len(noisy_wavs)):
                ID = f"{split}_{str(i).zfill(7)}"
                noisy_wav = noisy_wavs[i]
                clean_wav = clean_wavs[i]
                info = sb.dataio.dataio.read_audio_info(
                    noisy_wav.replace("$DATA_ROOT", data_folder)
                )
                duration = info.num_frames / info.sample_rate
                entry = dict(zip(headers, [ID, duration, noisy_wav, clean_wav]))
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
    takes = ["clean_wav", "noisy_wav"]
    provides = ["in_sig", "out_sig"]

    def audio_pipeline(clean_wav, noisy_wav):
        # Clean signal
        original_sample_rate = sb.dataio.dataio.read_audio_info(clean_wav).sample_rate
        clean_sig = sb.dataio.dataio.read_audio(clean_wav)

        # Noisy signal
        assert (
            original_sample_rate
            == sb.dataio.dataio.read_audio_info(noisy_wav).sample_rate
        )
        noisy_sig = sb.dataio.dataio.read_audio(noisy_wav)

        in_sig = torchaudio.functional.resample(
            noisy_sig,
            original_sample_rate,
            sample_rate,
        )
        yield in_sig

        out_sig = torchaudio.functional.resample(
            clean_sig,
            original_sample_rate,
            sample_rate,
        )
        yield out_sig

    sb.dataio.dataset.add_dynamic_item(
        [x for x in datasets if x is not None], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(
        [x for x in datasets if x is not None], ["id"] + provides
    )

    return datasets
