from typing import Union, List, Optional
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
import os

def combine_dataset(
    dataset_mixer: Union[dict, list],
    splits: List[str],
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = False,
    save_data_dir: Optional[str] = None,
    keep_ids: bool = False,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in
            all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `False`):
            Whether to shuffle the training and testing/validation data.
        save_data_dir (Optional[str], *optional*, defaults to `None`):
            Optional directory to save training/test mixes on.
        keep_ids (`bool`, *optional*, defaults to `False`):
            Whether to keep ids for training that are added during mixing.
            Used primarily in mix_data.py for saving, or the saved dataset has IDs already.
    """
    assert len(splits) == len(dataset_mixer), "Number of splits must match the number of datasets."
    if isinstance(dataset_mixer, list):
        assert len(dataset_mixer) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer}"
        mixer_dict = {}
        i = 0
        while i < len(dataset_mixer) - 1:
            assert isinstance(dataset_mixer[i], str), f"Invalid type in data mixer: {dataset_mixer}"
            if "." in dataset_mixer[i + 1]:
                value = float(dataset_mixer[i + 1])
            else:
                value = int(dataset_mixer[i + 1])
            mixer_dict[dataset_mixer[i]] = value
            i += 2
        dataset_mixer = mixer_dict

    if any(frac_or_samples < 0 for frac_or_samples in dataset_mixer.values()):
        raise ValueError("Dataset fractions / lengths cannot be negative.")

    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    # print save location
    if save_data_dir:
        print(f"Saving mixed dataset to {save_data_dir}")

    datasets = []
    for (ds, frac_or_samples), ds_config, split in zip(dataset_mixer.items(), configs, splits):
        # if dataset ends with .json, .jsonl, or .parquet, load from file
        if ds.endswith(".json") or ds.endswith(".jsonl"):
            dataset = load_dataset("json", data_files={split: os.path.join(ds, f"{split}.json")}, split=split)
        elif ds.endswith(".parquet") or os.path.isdir(ds):
            if os.path.isdir(ds):
                data_files = {split: os.path.join(ds, f"{split}.parquet")}
            else:
                data_files = {split: ds}
            dataset = load_dataset("parquet", data_files=data_files, split=split)
        else:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)
            except Exception:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

        # shuffle dataset if set
        if shuffle:
            dataset = dataset.shuffle(seed=42)

        # select a fraction of the dataset
        if frac_or_samples > 1.0:
            samples = int(frac_or_samples)
        else:
            samples = int(frac_or_samples * len(dataset))
        dataset = dataset.select(range(samples))

        # if id not in dataset, create it as ds-{index}
        if "id" not in dataset.column_names:
            id_col = [f"{ds}_{i}_{split}" for i in range(len(dataset))]
            dataset = dataset.add_column("id", id_col)

        # TODO: commenting out because of massive headache of specifying columns to keep
        # # Remove redundant columns to avoid schema conflicts on load
        # dataset = dataset.remove_columns(
        #     [col for col in dataset.column_names if col not in (columns_to_keep + ["id"])]
        # )
        datasets.append(dataset)

    datasets = concatenate_datasets(datasets)

    # optional save
    if save_data_dir:
        datasets.to_json(save_data_dir + "mixed_ds.json")

    if not keep_ids:
        # remove id column
        if "id" in datasets.column_names:
            datasets = datasets.remove_columns("id")

    return datasets
