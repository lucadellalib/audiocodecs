# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""LibriSpeech dataset."""

import csv
import logging
import os
import random
from collections import defaultdict
from typing import Optional, Sequence

import speechbrain as sb
import torch
import torchaudio
from speechbrain.dataio.dataio import merge_csvs


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
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ),
    ratios: Optional[Sequence[float]] = None,
    **kwargs,
) -> "None":
    """Prepare data manifest CSV files for the LibriSpeech dataset
    (see https://www.openslr.org/12).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    splits:
        The dataset splits to prepare.
    ratios:
        The list of ratios for splitting the data, based on speakers.
        If None, do not split.

    Raises
    ------
    RuntimeError
        If one of the expected split folders is missing.

    Examples
    --------
    >>> # Expected folder structure: LibriSpeech/{train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other}
    >>> prepare_data("LibriSpeech")

    """
    if not save_folder:
        save_folder = data_folder

    if ratios is not None and sum(ratios) > 1.0:
        raise ValueError(
            f"The sum of the provided split ratios exceeds 1.0: {sum(ratios)}"
        )

    # Write output CSV for each split
    for split in splits:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        # Files
        folder = os.path.join(data_folder, split)
        if not os.path.exists(folder):
            raise RuntimeError(f"{folder} does not exist")

        files = sorted(
            os.path.join(subfolder, x)
            for subfolder, _, files in os.walk(folder)
            for x in files
        )
        wavs = [
            x.replace(folder, os.path.join("$DATA_ROOT", split))
            for x in files
            if x.endswith(".wav") or x.endswith(".flac")
        ]

        # Read transcriptions
        trans = [x for x in files if x.endswith("trans.txt")]
        utt2twrd = {}
        for tran in trans:
            with open(tran, encoding="utf-8") as f:
                for line in f:
                    utt_id, wrd = line.split(" ", 1)
                    wrd = wrd.strip()
                    assert utt_id not in utt2twrd, (utt_id, wrd, tran)
                    utt2twrd[utt_id] = wrd
        wrds = []
        for wav in wavs:
            utt_id = os.path.splitext(os.path.basename(wav))[0]
            wrds.append(utt2twrd[utt_id])

        # Extract speaker ID (structure: /speaker_id/chapter_id/utterance_id)
        spk_ids = [
            os.path.basename(os.path.dirname(os.path.dirname(wav))) for wav in wavs
        ]

        if ratios is None:
            headers = ["ID", "duration", "wav", "wrd", "spk_id"]
            output_csv = os.path.join(save_folder, f"{split}.csv")
            _LOGGER.info(f"Writing {output_csv}...")
            with open(output_csv, "w", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(f, fieldnames=headers)
                csv_writer.writeheader()
                for i, (wav, wrd, spk_id) in enumerate(zip(wavs, wrds, spk_ids)):
                    ID = f"{split}_{str(i).zfill(7)}"
                    info = sb.dataio.dataio.read_audio_info(
                        wav.replace("$DATA_ROOT", data_folder)
                    )
                    duration = info.num_frames / info.sample_rate
                    entry = dict(zip(headers, [ID, duration, wav, wrd, spk_id]))
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
                headers = ["ID", "duration", "wav", "wrd", "spk_id"]
                output_csv = os.path.join(save_folder, f"{split}_{ratio}_{suffix}.csv")
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
                            ID = f"{split}_{str(i).zfill(7)}"
                            info = sb.dataio.dataio.read_audio_info(
                                wav.replace("$DATA_ROOT", data_folder)
                            )
                            duration = info.num_frames / info.sample_rate
                            entry = dict(zip(headers, [ID, duration, wav, wrd, spk_id]))
                            csv_writer.writerow(entry)
                            i += 1

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info("Done!")


def dataio_prepare(
    data_folder,
    train_csv,
    valid_csv,
    test_csv,
    sample_rate=16000,
    train_remove_if_longer=60.0,
    valid_remove_if_longer=60.0,
    test_remove_if_longer=60.0,
    sorting="ascending",
    debug=False,
    tokenizer=None,
    label_encoder=None,
    **kwargs,
):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    """
    if isinstance(train_csv, (list, tuple)):
        csvs = [os.path.basename(x) for x in train_csv]
        save_folder = os.path.dirname(train_csv[0])
        merge_csvs(
            save_folder,
            csvs,
            "train.csv",
        )
        train_csv = os.path.join(save_folder, "train.csv")

    if isinstance(valid_csv, (list, tuple)):
        csvs = [os.path.basename(x) for x in valid_csv]
        save_folder = os.path.dirname(valid_csv[0])
        merge_csvs(
            save_folder,
            csvs,
            "valid.csv",
        )
        valid_csv = os.path.join(save_folder, "valid.csv")

    if isinstance(test_csv, (list, tuple)):
        csvs = [os.path.basename(x) for x in test_csv]
        save_folder = os.path.dirname(test_csv[0])
        merge_csvs(
            save_folder,
            csvs,
            "test.csv",
        )
        test_csv = os.path.join(save_folder, "test.csv")

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv,
        replacements={"DATA_ROOT": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=sorting == "descending",
        key_max_value={"duration": train_remove_if_longer},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_csv,
        replacements={"DATA_ROOT": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": valid_remove_if_longer},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=test_csv,
        replacements={"DATA_ROOT": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": test_remove_if_longer},
    )

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = ["wav", "wrd", "spk_id"]
    provides = ["sig", "toks", "wrd", "utt_label"]

    def audio_pipeline(wav, wrd, spk_id):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(sig, original_sample_rate, sample_rate)
        yield sig

        if tokenizer is None:
            yield None
        else:
            toks = torch.LongTensor(tokenizer.sp.encode_as_ids(wrd))
            yield toks

        yield wrd

        if label_encoder is None:
            yield None
        else:
            utt_label = label_encoder.encode_sequence_torch([spk_id])
            yield utt_label

    sb.dataio.dataset.add_dynamic_item(
        [train_data, valid_data, test_data], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return train_data, valid_data, test_data
