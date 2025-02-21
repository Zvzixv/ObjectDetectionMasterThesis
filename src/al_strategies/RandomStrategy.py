import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.al_strategies.strategy import AbstractStrategy


class RandomStrategy(AbstractStrategy):

    def get_next_dataset(self, model_type: str, training_data: DataLoader,
                         unlabeled_data: DataLoader,
                         n_samples_to_add: int,
                         batch_size: int,
                         image_data) -> tuple[DataLoader, DataLoader, any]:
        n_train = len(training_data.dataset)
        n = len(unlabeled_data.dataset)
        assert n >= n_samples_to_add, "Not enough data to add"
        indices = np.random.choice(np.arange(n), n_samples_to_add,
                                   replace=False)
        assert len(indices) == n_samples_to_add
        train_data_loader, complement_data_loader = self.get_subsets(
            indices=indices, unlabeled_data=unlabeled_data,
            training_data=training_data, batch_size=batch_size)
        assert len(train_data_loader.dataset) + len(
            complement_data_loader.dataset) == n + n_train, f"{len(train_data_loader)} + {len(complement_data_loader)} == {n} + {n_train}"

        if model_type == "yolo":
            predictions = [
                (None, image_name) for image_name in
                os.listdir(Path(
                    self.yolo_data_path) / "active_learning_unlabeled" / "images")]
            best_images_names = [pred[1] for idx, pred in enumerate(predictions)
                                 if
                                 idx in indices]
            assert len(best_images_names) == n_samples_to_add

            image_ids = [image_sample["id"] for image_sample in image_data if
                         image_sample["file_name"] in best_images_names]

        else:
            image_ids = []
        return train_data_loader, complement_data_loader, image_ids
