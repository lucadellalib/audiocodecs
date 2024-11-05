# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Common utilities."""

import importlib
import os

from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import run_on_main


__all__ = ["prepare_recipe"]


_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def prepare_recipe(hparams, run_opts):
    # Dataset preparation
    dataset_module = import_module(
        os.path.join(_ROOT_DIR, "datasets", f"{hparams['dataset'].lower()}.py")
    )

    prepare_data = dataset_module.prepare_data
    dataio_prepare = dataset_module.dataio_prepare

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(prepare_data, kwargs=hparams)

    # Fit tokenizer
    if "tokenizer_cls" in hparams:
        hparams["tokenizer"] = hparams["tokenizer_cls"]()

    # Create the datasets objects
    train_data, valid_data, test_data = dataio_prepare(
        debug=run_opts.get("debug", False), **hparams
    )

    # Fit label encoder
    if "label_encoder" in hparams:
        hparams["label_encoder"].load_or_create(
            path=os.path.join(hparams["save_folder"], "label_encoder.txt"),
            from_didatasets=[train_data],
            output_key=hparams.get("output_key", None),
        )

    # Dynamic batching
    hparams["train_dataloader_kwargs"] = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "train_dynamic_batching", False
    ):
        hparams["train_dataloader_kwargs"]["batch_sampler"] = DynamicBatchSampler(
            train_data,
            hparams["train_max_batch_length"],
            num_buckets=hparams.get("num_buckets"),
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering=hparams.get("sorting", "batch_ordering"),
            max_batch_ex=hparams.get("max_batch_size"),
            bucket_boundaries=hparams.get("bucket_boundaries", []),
            lengths_list=hparams.get("lengths_list"),
        )
    else:
        hparams["train_dataloader_kwargs"]["batch_size"] = hparams["train_batch_size"]
        hparams["train_dataloader_kwargs"]["shuffle"] = hparams["sorting"] == "random"
        hparams["train_dataloader_kwargs"]["pin_memory"] = (
            run_opts.get("device", "cpu") != "cpu"
        )
        hparams["train_dataloader_kwargs"]["drop_last"] = hparams.get(
            "segment_size", None
        ) is not None and hparams.get("segment_pad", False)

    hparams["valid_dataloader_kwargs"] = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "valid_dynamic_batching", False
    ):
        hparams["valid_dataloader_kwargs"]["batch_sampler"] = DynamicBatchSampler(
            valid_data,
            hparams["valid_max_batch_length"],
            num_buckets=hparams.get("num_buckets"),
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams.get("max_batch_size"),
            bucket_boundaries=hparams.get("bucket_boundaries", []),
            lengths_list=hparams.get("lengths_list"),
        )
    else:
        hparams["valid_dataloader_kwargs"]["batch_size"] = hparams["valid_batch_size"]
        hparams["valid_dataloader_kwargs"]["pin_memory"] = (
            run_opts.get("device", "cpu") != "cpu"
        )

    hparams["test_dataloader_kwargs"] = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "test_dynamic_batching", False
    ):
        hparams["test_dataloader_kwargs"]["batch_sampler"] = DynamicBatchSampler(
            test_data,
            hparams["test_max_batch_length"],
            num_buckets=hparams.get("num_buckets"),
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams.get("max_batch_size"),
            bucket_boundaries=hparams.get("bucket_boundaries", []),
            lengths_list=hparams.get("lengths_list"),
        )
    else:
        hparams["test_dataloader_kwargs"]["batch_size"] = hparams["test_batch_size"]
        hparams["test_dataloader_kwargs"]["pin_memory"] = (
            run_opts.get("device", "cpu") != "cpu"
        )

    # Pretrain the specified modules
    if "pretrainer" in hparams:
        run_on_main(hparams["pretrainer"].collect_files)
        run_on_main(hparams["pretrainer"].load_collected)

    return hparams, train_data, valid_data, test_data


def import_module(path):
    """Import a Python module at runtime.

    Parameters
    ----------
    path:
        The absolute or relative path to the module.

    Returns
    -------
        The imported module.

    """
    path = os.path.realpath(path)
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
