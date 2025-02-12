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

"""VCTK dataset."""

import csv
import logging
import os
from collections import defaultdict
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

# fmt: off
_TRAIN_SPEAKER_IDS = [
    "p279", "p333", "p323", "p228", "p225", "p231", "p269", "p258", "p281", "p266",
    "p244", "p282", "p376", "p286", "p307", "p240", "p310", "p263", "p276", "p259",
    "p249", "p312", "p340", "p275", "p295", "p234", "p335", "p311", "p301", "p294",
    "p248", "p237", "p257", "p278", "p374", "p267", "p336", "p305", "p247", "p245",
    "p273", "p299", "p283", "p298", "p345", "p277", "p284", "p343", "p304", "p314",
    "p287", "p254", "p326", "p261", "p270", "p262", "p253", "p292", "p236", "p334",
    "p264", "p300", "p313", "p246", "p341", "p360", "p293", "p306", "p255", "p308",
    "p317", "p303", "p361", "p250", "p251", "p241", "p362", "p226", "p297", "p288",
    "p363", "p230", "p364", "p280", "p256", "p351", "p339", "p265", "p271", "p315",
    "p272", "p233", "p239", "p302", "p330", "p238", "p243", "p274", "p232", "p329",
    "p318", "p316", "p260", "p285", "p268", "p227", "p347", "p252", "p229", "s5",
]
# fmt: on


def prepare_data(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = ("train", "valid", "test"),
    num_valid_speakers: "int" = 2,
    num_test_speakers: "int" = 8,
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the VCTK dataset.

    The following files must be downloaded from the official website (https://datashare.ed.ac.uk/handle/10283/3443)
    and extracted to a common folder (e.g. `VCTK`):
    - `VCTK-Corpus-0.92.zip`

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
    num_test_speakers:
        The number of speakers in the training set to use for test
        (these speakers will be removed from the training set).

    Raises
    ------
    ValueError
        If an invalid argument value is given.
    RuntimeError
        If an invalid split is given.

    Examples
    --------
    >>> # Expected folder structure:
    >>> # VCTK/{txt, wav48_silence_trimmed, speaker-info.txt, update.txt}
    >>> prepare_data("VCTK")

    """
    if not save_folder:
        save_folder = data_folder
    os.makedirs(save_folder, exist_ok=True)

    if num_valid_speakers > len(_TRAIN_SPEAKER_IDS):
        raise ValueError(
            f"`num_valid_speakers` ({num_valid_speakers}) must be <= than the total "
            f"number of speakers in the training set ({len(_TRAIN_SPEAKER_IDS)})"
        )
    if num_test_speakers > (len(_TRAIN_SPEAKER_IDS) - num_valid_speakers):
        raise ValueError(
            f"`num_test_speakers` ({num_test_speakers}) must be <= than the total "
            f"number of speakers in the training set after removing the validation "
            f"speakers ({(len(_TRAIN_SPEAKER_IDS) - num_valid_speakers)})"
        )

    # Files
    folder = os.path.join(data_folder, "wav48_silence_trimmed")
    if not os.path.exists(folder):
        raise RuntimeError(f"{folder} does not exist")

    all_wavs = sorted(
        os.path.join(subfolder, x)
        for subfolder, _, files in os.walk(folder)
        for x in files
    )
    all_wavs = [
        x.replace(folder, os.path.join("$DATA_ROOT", "wav48_silence_trimmed"))
        for x in all_wavs
        if x.endswith(".wav") or x.endswith(".flac")
    ]

    # Write output CSV for each split
    for split in splits:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        if split == "train":
            spk_ids = _TRAIN_SPEAKER_IDS[num_valid_speakers + num_test_speakers :]
        elif split == "valid":
            spk_ids = _TRAIN_SPEAKER_IDS[:num_valid_speakers]
        elif split == "test":
            spk_ids = _TRAIN_SPEAKER_IDS[
                num_valid_speakers : num_valid_speakers + num_test_speakers
            ]
        else:
            raise RuntimeError(f"{split} is not a valid split")

        wavs = []
        spk2utt = defaultdict(list)
        utt2spk = defaultdict(list)
        for wav in all_wavs:
            spk_id, utt_id, mic_id = os.path.splitext(os.path.basename(wav))[0].split(
                "_"
            )
            if (
                int(utt_id) < 25  # Only the first 24 utterances are parallel
                and mic_id == "mic1"  # Not much difference between mics
                and spk_id in spk_ids
            ):
                wavs.append(wav)
                spk2utt[spk_id].append((utt_id, wav))
                utt2spk[utt_id].append((spk_id, wav))
                del spk_id, utt_id, mic_id

        # Build parallel data in a deterministic way
        src_wavs = []
        tgt_wavs = []
        spk_wavs = []
        for utt_id, spk_ids_wavs in utt2spk.items():
            for i in range(len(spk_ids_wavs) - 1):
                src_spk_id, src_wav = spk_ids_wavs[i]
                tgt_spk_id, tgt_wav = spk_ids_wavs[i + 1]
                utt_ids_wavs = spk2utt[tgt_spk_id]
                tmp = []
                for j in range(len(utt_ids_wavs)):
                    spk_utt_id, spk_wav = utt_ids_wavs[j]
                    if spk_utt_id != utt_id:
                        tmp.append(spk_wav)
                src_wavs.append(src_wav)
                tgt_wavs.append(tgt_wav)
                spk_wavs.append(tmp)

        headers = ["ID", "duration", "src_wav", "tgt_wav", "spk_wavs"]
        output_csv = os.path.join(save_folder, f"{split}.csv")
        _LOGGER.info(f"Writing {output_csv}...")
        with open(output_csv, "w", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            for i in range(len(src_wavs)):
                ID = f"{split}_{str(i).zfill(7)}"
                src_wav = src_wavs[i]
                tgt_wav = tgt_wavs[i]
                spk_wav = spk_wavs[i]
                src_info = sb.dataio.dataio.read_audio_info(
                    src_wav.replace("$DATA_ROOT", data_folder)
                )
                src_duration = src_info.num_frames / src_info.sample_rate
                entry = dict(
                    zip(
                        headers,
                        [ID, src_duration, src_wav, tgt_wav, ", ".join(spk_wav)],
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
    takes = ["src_wav", "tgt_wav", "spk_wavs"]
    provides = ["in_sig", "out_sig", "spk_sigs"]

    def audio_pipeline(src_wav, tgt_wav, spk_wavs):
        # Source signal
        original_sample_rate = sb.dataio.dataio.read_audio_info(src_wav).sample_rate
        src_sig = sb.dataio.dataio.read_audio(src_wav)

        # Target signal
        assert (
            original_sample_rate
            == sb.dataio.dataio.read_audio_info(tgt_wav).sample_rate
        )
        tgt_sig = sb.dataio.dataio.read_audio(tgt_wav)

        # Speaker signals
        spk_sigs = []
        for spk_wav in spk_wavs.split(", "):
            assert (
                original_sample_rate
                == sb.dataio.dataio.read_audio_info(spk_wav).sample_rate
            )
            spk_sig = sb.dataio.dataio.read_audio(spk_wav)
            spk_sig = torchaudio.functional.resample(
                spk_sig,
                original_sample_rate,
                sample_rate,
            )
            spk_sigs.append(spk_sig)

        in_sig = torchaudio.functional.resample(
            src_sig,
            original_sample_rate,
            sample_rate,
        )

        out_sig = torchaudio.functional.resample(
            tgt_sig,
            original_sample_rate,
            sample_rate,
        )

        yield in_sig
        yield out_sig
        yield spk_sigs

    sb.dataio.dataset.add_dynamic_item(
        [x for x in datasets if x is not None], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(
        [x for x in datasets if x is not None], ["id"] + provides
    )

    return datasets
