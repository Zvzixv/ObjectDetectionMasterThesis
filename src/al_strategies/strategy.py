import os
from pathlib import Path

import numpy as np
import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset

from src.utils import collate_fn


def update_active_training_set_for_yolo(image_data: list[dict],
                                        image_ids: list[int],
                                        data_path: str | Path,
                                        logger):
    n_unlabeled_before = len(
        os.listdir(Path(data_path) / "active_learning_unlabeled" / "images"))

    train_files_before = os.listdir(
        Path(data_path) / "active_learning_train" / "images")
    n_train_before = len(train_files_before)
    n_points_to_add = len(image_ids)

    # Wyznacza nazwy plikow labels do skopiowania
    filenames_to_move = [image["file_name"] for image in image_data if
                         image["id"] in image_ids]

    assert len(list(set(filenames_to_move))) == n_points_to_add
    # kopiuje pliki
    logger.info("Moving selected unlabeled samples to labeled samples")
    for filename in tqdm.tqdm(filenames_to_move):
        os.rename(
            Path(data_path) / "active_learning_unlabeled" / "images" / filename,
            Path(data_path) / "active_learning_train" / "images" / filename)
        assert os.path.exists(
            Path(data_path) / "active_learning_train" / "images" / filename)

    n_unlabeled_after = len(os.listdir(
        Path(data_path) / "active_learning_unlabeled" / "images"))
    n_train_after = len(
        os.listdir(Path(
            data_path) / "active_learning_train" / "images"))

    assert n_unlabeled_before == n_unlabeled_after + n_points_to_add, f"{n_unlabeled_before} != {n_unlabeled_after + n_points_to_add}, {missing_files()}"
    assert n_train_before + n_points_to_add == n_train_after, f"{n_train_before + n_points_to_add} != {n_train_after}, {missing_files()}"


class AbstractStrategy:
    def __init__(self, yolo_data_path: str, logger=None):
        self.yolo_data_path = yolo_data_path
        self.logger = logger

    def get_next_dataset(self, *args, **kwargs) -> tuple[
        DataLoader, DataLoader, any]:
        raise NotImplementedError

    def get_subsets(self, indices: list[int], training_data, unlabeled_data,
                    batch_size):
        # 18410
        ds = unlabeled_data.dataset
        assert max(indices) <= len(ds), f"{indices}"
        # 1315
        ds_subset = Subset(ds, indices)
        assert len(ds_subset) == len(indices)
        # 18408
        remaining_labels = np.setdiff1d(np.arange(len(ds)), indices)
        assert len(remaining_labels) + len(indices) == len(ds)
        assert len(np.intersect1d(remaining_labels, indices)) == 0
        unlabeled_remaining_subset = Subset(ds, remaining_labels)
        assert len(unlabeled_remaining_subset) == len(ds) - len(
            indices), f"{len(unlabeled_remaining_subset)} = {len(ds)} - {len(indices)}"
        previous_training_subset = training_data.dataset

        new_training_ds = ConcatDataset([ds_subset, previous_training_subset])
        assert len(new_training_ds) == len(ds_subset) + len(
            previous_training_subset)
        self.logger.info(f"new_training_ds:   {len(new_training_ds)}")

        return DataLoader(new_training_ds, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn), DataLoader(
            unlabeled_remaining_subset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn)
