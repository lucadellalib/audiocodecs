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

"""Speech Commands dataset."""

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

# Constants matching the official split logic
_HASH_DIVIDER = "_nohash_"

_EXCEPT_FOLDER = "_background_noise_"


def prepare_data(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = ("training", "validation", "testing"),
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the Speech Commands dataset
    (see https://www.tensorflow.org/datasets/catalog/speech_commands).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    splits:
        The dataset splits to prepare.

    Raises
    ------
    RuntimeError
        If one of the expected manifest files is missing.

    Examples
    --------
    >>> # Expected folder structure: SpeechCommands/{_background_noise_, backward, bed, bird, cat, ..., validation_list.txt}
    >>> prepare_data("SpeechCommands")

    """
    if not save_folder:
        save_folder = data_folder
    os.makedirs(save_folder, exist_ok=True)

    want_val = "validation" in splits
    want_test = "testing" in splits

    # Check lists existence if requested
    val_list_path = os.path.join(data_folder, "validation_list.txt")
    test_list_path = os.path.join(data_folder, "testing_list.txt")
    if want_val and not os.path.isfile(val_list_path):
        raise RuntimeError(f"{val_list_path} does not exist")
    if want_test and not os.path.isfile(test_list_path):
        raise RuntimeError(f"{test_list_path} does not exist")

    # Build walkers for each split
    walkers = {}

    if "validation" in splits:
        items = []
        with open(val_list_path, "r", encoding="utf-8") as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                items.append(os.path.normpath(os.path.join(data_folder, rel)))
        walkers["validation"] = items

    if "testing" in splits:
        items = []
        with open(test_list_path, "r", encoding="utf-8") as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                items.append(os.path.normpath(os.path.join(data_folder, rel)))
        walkers["testing"] = items

    if "training" in splits:
        # All candidate wavs: class_subdir/*.wav
        all_wavs = []
        for root, _, files in os.walk(data_folder):
            for fname in files:
                if fname.endswith(".wav"):
                    all_wavs.append(os.path.join(root, fname))
        all_wavs = sorted(all_wavs)
        # Exclude background noise and keep only files with HASH_DIVIDER
        all_wavs = [
            w
            for w in all_wavs
            if (_EXCEPT_FOLDER not in w and _HASH_DIVIDER in os.path.basename(w))
        ]

        # Exclude validation + testing items (normalize paths for safety)
        excludes = set()
        if os.path.isfile(val_list_path):
            with open(val_list_path, "r", encoding="utf-8") as f:
                for line in f:
                    rel = line.strip()
                    if rel:
                        excludes.add(os.path.normpath(os.path.join(data_folder, rel)))
        if os.path.isfile(test_list_path):
            with open(test_list_path, "r", encoding="utf-8") as f:
                for line in f:
                    rel = line.strip()
                    if rel:
                        excludes.add(os.path.normpath(os.path.join(data_folder, rel)))

        walkers["training"] = [
            w for w in all_wavs if os.path.normpath(w) not in excludes
        ]

    # Write CSVs
    for split in splits:
        wav_list = walkers.get(split, [])
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        headers = ["ID", "duration", "wav", "command"]
        output_csv = os.path.join(save_folder, f"{split}.csv")
        _LOGGER.info(f"Writing {output_csv}...")

        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for i, abs_wav in enumerate(wav_list):
                rel_wav = abs_wav.replace(data_folder, "$DATA_ROOT")
                command = os.path.basename(os.path.dirname(abs_wav))

                info = sb.dataio.dataio.read_audio_info(abs_wav)
                duration = info.num_frames / info.sample_rate

                writer.writerow(
                    {
                        "ID": f"{split}_{str(i).zfill(7)}",
                        "duration": duration,
                        "wav": rel_wav,
                        "command": command,
                    }
                )

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
    takes = ["wav", "command"]
    provides = ["sig", "utt_label"]

    def audio_pipeline(wav, command):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(sig, original_sample_rate, sample_rate)
        yield sig

        if label_encoder is None:
            yield os.path.basename(wav).replace(".flac", ".wav")
        else:
            utt_label = label_encoder.encode_sequence_torch([command])
            yield utt_label

    sb.dataio.dataset.add_dynamic_item(
        [x for x in datasets if x is not None], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(
        [x for x in datasets if x is not None], ["id"] + provides
    )

    return datasets
