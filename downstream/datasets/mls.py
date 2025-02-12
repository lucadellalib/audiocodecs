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

"""Multilingual LibriSpeech (MLS) dataset."""

import csv
import logging
import os
import random
from collections import defaultdict
from typing import Optional, Sequence

import speechbrain as sb
import torch
import torchaudio


__all__ = ["dataio_prepare", "prepare_data"]


# Seed
random.seed(0)

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


_LANG_MAP = {
    "dutch": "nl",
    "french": "fr",
    "german": "de",
    "english": "en",
    "italian": "it",
    "spanish": "es",
    "portuguese": "pt",
    "polish": "pl",
}


def prepare_data(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = ("train", "dev", "test"),
    languages: "Optional[List[str]]" = None,
    ratios: "Optional[Sequence[float]]" = None,
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the Multilingual LibriSpeech (MLS) dataset
    (see https://www.openslr.org/94).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    splits:
        The dataset splits to prepare.
    languages:
        A list of specific languages to process. If None, process all languages.
    ratios:
        The list of ratios for splitting the data, based on speakers.
        If None, do not split.

    Raises
    ------
    RuntimeError
        If one of the expected split folders is missing.

    Examples
    --------
    >>> # Expected folder structure: MLS/{mls_dutch, mls_french, ...}/{train, dev, test}
    >>> prepare_data("MLS", languages=["dutch", "french"])
    """
    if not save_folder:
        save_folder = data_folder
    os.makedirs(save_folder, exist_ok=True)

    if ratios is not None and sum(ratios) > 1.0:
        raise ValueError(
            f"The sum of the provided split ratios exceeds 1.0: {sum(ratios)}"
        )

    # List all available language directories
    all_lang_dirs = [
        lang_dir
        for lang_dir in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, lang_dir))
    ]

    # Filter languages if a specific list is provided
    if languages:
        lang_dirs = [
            lang_dir
            for lang_dir in all_lang_dirs
            if lang_dir.split("_")[-1] in languages
        ]
        missing_languages = set([f"mls_{x}" for x in languages]) - set(lang_dirs)
        if missing_languages:
            _LOGGER.warning(f"Missing specified languages: {missing_languages}")
    else:
        lang_dirs = all_lang_dirs
        languages = [lang_dir.split("_")[-1] for lang_dir in lang_dirs]

    if not lang_dirs:
        raise RuntimeError("No valid language directories found to process")

    # Iterate over each language directory
    for lang_dir in lang_dirs:
        lang = lang_dir.split("_")[-1]
        lang_path = os.path.join(data_folder, lang_dir)
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Language: {lang}")

        # Process each split
        for split in splits:
            split_path = os.path.join(lang_path, split)
            if not os.path.exists(split_path):
                raise RuntimeError(f"Missing split: {split_path}. Skipping...")

            _LOGGER.info(
                "----------------------------------------------------------------------",
            )
            _LOGGER.info(f"Split: {split}")

            # Collect files
            files = sorted(
                os.path.join(subfolder, x)
                for subfolder, _, files in os.walk(split_path)
                for x in files
            )
            wavs = [
                x.replace(split_path, os.path.join("$DATA_ROOT", lang_dir, split))
                for x in files
                if x.endswith(".wav") or x.endswith(".flac")
            ]

            # Read transcriptions
            trans = [x for x in files if x.endswith("transcripts.txt")]
            utt2twrd = {}
            for tran in trans:
                with open(tran, encoding="utf-8") as f:
                    for line in f:
                        utt_id, wrd = line.split("\t", 1)
                        wrd = wrd.strip()
                        assert utt_id not in utt2twrd, (utt_id, wrd, tran)
                        utt2twrd[utt_id] = wrd
            wrds = []
            for wav in wavs:
                utt_id = os.path.splitext(os.path.basename(wav))[0]
                wrds.append(utt2twrd[utt_id])

            # Extract speaker ID
            spk_ids = [
                os.path.basename(os.path.dirname(os.path.dirname(wav))) for wav in wavs
            ]

            if ratios is None:
                headers = ["ID", "duration", "wav", "wrd", "spk_id", "lang"]
                output_csv = os.path.join(save_folder, f"{split}_{lang}.csv")
                _LOGGER.info(f"Writing {output_csv}...")
                with open(output_csv, "w", encoding="utf-8") as f:
                    csv_writer = csv.DictWriter(f, fieldnames=headers)
                    csv_writer.writeheader()
                    for i, (wav, wrd, spk_id) in enumerate(zip(wavs, wrds, spk_ids)):
                        ID = f"{split}_{lang}_{str(i).zfill(7)}"
                        info = sb.dataio.dataio.read_audio_info(
                            wav.replace("$DATA_ROOT", data_folder)
                        )
                        duration = info.num_frames / info.sample_rate
                        entry = dict(
                            zip(headers, [ID, duration, wav, wrd, spk_id, lang])
                        )
                        csv_writer.writerow(entry)
            else:
                # Group data by speaker ID
                spk2data = defaultdict(list)
                for wav, wrd, spk_id in zip(wavs, wrds, spk_ids):
                    spk2data[spk_id].append((wav, wrd))

                unique_spks = list(spk2data.keys())

                # Shuffle speakers for randomization
                random.shuffle(unique_spks)

                # Generate suffixes to avoid file name collisions (a, b, c, etc.)
                suffixes = [chr(i) for i in range(97, 123)]  # Generate 'a' to 'z'

                # Calculate the number of utterances per speaker per split (except for the last ratio)
                num_utts_per_spk = [
                    [int(ratio * len(spk2data[spk_id])) for spk_id in unique_spks]
                    for ratio in ratios[:-1]
                ]

                # Ensure the remainder goes to the last split
                for i, spk_id in enumerate(unique_spks):
                    total_utts = len(spk2data[spk_id])
                    assigned_utts = sum(
                        num_utts_per_spk[j][i] for j in range(len(ratios) - 1)
                    )
                    remaining_utts = total_utts - assigned_utts
                    num_utts_per_spk.append([remaining_utts for _ in unique_spks])

                # Process each ratio to generate stratified splits
                for j, (ratio, suffix) in enumerate(zip(ratios, suffixes)):
                    headers = ["ID", "duration", "wav", "wrd", "spk_id", "lang"]
                    output_csv = os.path.join(
                        save_folder, f"{split}_{lang}_{ratio}_{suffix}.csv"
                    )
                    _LOGGER.info(f"Writing {output_csv} for ratio {ratio}...")

                    with open(output_csv, "w", encoding="utf-8") as f:
                        csv_writer = csv.DictWriter(f, fieldnames=headers)
                        csv_writer.writeheader()

                        # Process each speaker, keeping the same speakers but varying their proportions
                        i = 0
                        for spk_id, num_utts in zip(unique_spks, num_utts_per_spk[j]):
                            # Get speaker's utterances
                            utts = spk2data[spk_id]

                            # Select the utterances for this split
                            selected_utts = utts[:num_utts]

                            # Remove allocated utterances
                            spk2data[spk_id] = utts[num_utts:]

                            # Write selected utterances to the CSV
                            for wav, wrd in selected_utts:
                                ID = f"{split}_{lang}_{str(i).zfill(7)}"
                                info = sb.dataio.dataio.read_audio_info(
                                    wav.replace("$DATA_ROOT", data_folder)
                                )
                                duration = info.num_frames / info.sample_rate
                                entry = dict(
                                    zip(headers, [ID, duration, wav, wrd, spk_id, lang])
                                )
                                csv_writer.writerow(entry)
                                i += 1

    # Merge languages for each split
    for split in splits:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Merging manifests for split: {split}")
        csv_files = [
            os.path.join(save_folder, f)
            for f in os.listdir(save_folder)
            if (f.startswith(split) and any([x in f for x in languages]))
        ]

        merged_csv = os.path.join(save_folder, f"{split}.csv")

        # Read and merge all CSVs
        merged_data = []
        headers = None
        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if headers is None:
                    headers = reader.fieldnames  # Extract headers from the first file
                merged_data.extend(list(reader))

        # Write the merged data
        with open(merged_csv, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(merged_data)

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
    tokenizer=None,
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
    takes = ["wav", "wrd", "spk_id", "lang"]
    provides = ["sig", "toks", "wrd", "utt_label", "locale"]

    def audio_pipeline(wav, wrd, spk_id, lang):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(sig, original_sample_rate, sample_rate)
        yield sig

        if tokenizer is None:
            yield wrd
        else:
            if hasattr(tokenizer, "sp"):
                toks = tokenizer.sp.encode_as_ids(wrd)
            else:
                toks = tokenizer(wrd)
            toks = torch.LongTensor(toks)
            yield toks

        yield wrd

        if label_encoder is None:
            yield None
        else:
            utt_label = label_encoder.encode_sequence_torch([spk_id])
            yield utt_label

        yield _LANG_MAP[lang]

    sb.dataio.dataset.add_dynamic_item(
        [x for x in datasets if x is not None], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(
        [x for x in datasets if x is not None], ["id"] + provides
    )

    return datasets
