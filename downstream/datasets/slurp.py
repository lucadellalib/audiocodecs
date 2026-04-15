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

"""SLURP dataset."""

import csv
import json
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
    splits: "Sequence[str]" = ("train", "train_synthetic", "devel", "test"),
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the SLURP dataset
    (see https://github.com/pswietojanski/slurp).

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
    >>> # Expected folder structure: SLURP/{slurp_real, slurp_synth, devel.jsonl, test.jsonl, train.jsonl, train_synthetic.jsonl}
    >>> prepare_data("SLURP")

    """
    if not save_folder:
        save_folder = data_folder
    os.makedirs(save_folder, exist_ok=True)

    # Write output CSV for each split
    for split in splits:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        # Read annotations
        jsonl_path = os.path.join(data_folder, f"{split}.jsonl")
        if not os.path.isfile(jsonl_path):
            raise RuntimeError(f"{jsonl_path} does not exist")

        wavs = []
        scenarios = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    # Skip empty lines
                    continue
                obj = json.loads(line)
                scenario = obj["scenario"]
                for audio in obj["recordings"]:
                    prefix = "slurp_synth" if "synthetic" in split else "slurp_real"
                    wav = os.path.join("$DATA_ROOT", prefix, audio["file"])
                    wavs.append(wav)
                    scenarios.append(scenario)

        headers = ["ID", "duration", "wav", "scenario"]
        output_csv = os.path.join(save_folder, f"{split}.csv")
        _LOGGER.info(f"Writing {output_csv}...")
        with open(output_csv, "w", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            for i, (wav, scenario) in enumerate(zip(wavs, scenarios)):
                ID = f"{split}_{str(i).zfill(7)}"
                info = sb.dataio.dataio.read_audio_info(
                    wav.replace("$DATA_ROOT", data_folder)
                )
                duration = info.num_frames / info.sample_rate
                entry = dict(zip(headers, [ID, duration, wav, scenario]))
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
    takes = ["wav", "scenario"]
    provides = ["sig", "utt_label"]

    def audio_pipeline(wav, scenario):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(sig, original_sample_rate, sample_rate)
        yield sig

        if label_encoder is None:
            yield os.path.basename(wav).replace(".flac", ".wav")
        else:
            utt_label = label_encoder.encode_sequence_torch([scenario])
            yield utt_label

    sb.dataio.dataset.add_dynamic_item(
        [x for x in datasets if x is not None], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(
        [x for x in datasets if x is not None], ["id"] + provides
    )

    return datasets
