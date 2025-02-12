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

"""IEMOCAP dataset."""

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


def prepare_data(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = (
        "Session1",
        "Session2",
        "Session3",
        "Session4",
        "Session5F",
        "Session5M",
    ),
    emotions: "Sequence[str]" = (
        "sad",
        "hap",
        "fea",
        "sur",
        "oth",
        "xxx",
        "fru",
        "ang",
        "exc",
        "dis",
        "neu",
    ),
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the IEMOCAP dataset.

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    splits:
        The dataset splits to prepare.
        In the case of IEMOCAP, this refers to different sessions.
    emotions:
        The list of emotion labels to consider.
        Utterances that do not match these emotion labels will be dropped.

    Raises
    ------
    RuntimeError
        If one of the expected split folders is missing.

    Examples
    --------
    >>> # Expected folder structure: IEMOCAP/{Session1, Session2, Session3, Session4, Session5}
    >>> prepare_data("IEMOCAP")

    """
    if not save_folder:
        save_folder = data_folder
    os.makedirs(save_folder, exist_ok=True)

    # Write output CSV for each session
    for split in splits:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        # Session folder
        if split in ["Session5F", "Session5M"]:
            session_folder = os.path.join(data_folder, "Session5")
        else:
            session_folder = os.path.join(data_folder, split)
        if not os.path.exists(session_folder):
            raise RuntimeError(f"{session_folder} does not exist")

        # Collecting audio files and transcripts
        wavs = sorted(
            os.path.join(subfolder, file)
            for subfolder, _, files in os.walk(
                os.path.join(session_folder, "sentences", "wav")
            )
            for file in files
            if file.endswith(".wav")
        )

        # Read emotion labels
        utt2emo = {}
        emo_folder = os.path.join(session_folder, "dialog", "EmoEvaluation")
        for filename in os.listdir(emo_folder):
            if not filename.endswith(".txt"):
                continue
            emo_file = os.path.join(emo_folder, filename)
            with open(emo_file, encoding="utf-8") as f:
                for line in f:
                    # Only process lines that start with emotion info
                    if line.startswith("["):
                        parts = line.strip().split()
                        utt_id = parts[3]
                        emo = parts[4]
                        spk_id = utt_id.split("_")[-1]
                        if split == "Session5M" and spk_id.startswith("F"):
                            continue
                        if split == "Session5F" and spk_id.startswith("M"):
                            continue
                        if emo not in emotions:
                            continue
                        utt2emo[utt_id] = emo

        # Prepare entries for CSV
        headers = ["ID", "duration", "wav", "emo"]
        output_csv = os.path.join(save_folder, f"{split}.csv")
        _LOGGER.info(f"Writing {output_csv}...")
        with open(output_csv, "w", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            for i, wav in enumerate(wavs):
                utt_id = os.path.splitext(os.path.basename(wav))[0]
                if utt_id not in utt2emo:
                    continue  # skip files without emotion labels
                emo = utt2emo[utt_id]
                info = sb.dataio.dataio.read_audio_info(wav)
                duration = info.num_frames / info.sample_rate
                entry = dict(zip(headers, [utt_id, duration, wav, emo]))
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
    label_encoder=None,
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
    takes = ["wav", "emo"]
    provides = ["sig", "utt_label"]

    def audio_pipeline(wav, spk_id):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(sig, original_sample_rate, sample_rate)
        yield sig

        if label_encoder is None:
            yield None
        else:
            utt_label = label_encoder.encode_sequence_torch([spk_id])
            yield utt_label

    sb.dataio.dataset.add_dynamic_item(
        [x for x in datasets if x is not None], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(
        [x for x in datasets if x is not None], ["id"] + provides
    )

    return datasets
