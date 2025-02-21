import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.al_strategies.strategy import AbstractStrategy
from src.models.rcnn_model import RCNN
from src.models.yolov5_model import YOLOTrainer


class MinMaxStrategy(AbstractStrategy):
    def get_next_dataset(self, model: YOLOTrainer | torch.nn.Module,
                         training_data: DataLoader,
                         unlabeled_data: DataLoader,
                         n_samples_to_add: int,
                         image_data: dict,
                         batch_size: int, before_last_training: bool,
                         yolo_model_path: str | Path = None,
                         pred_save_dir: str | Path = None
                         ) -> tuple[
        DataLoader, DataLoader, any]:

        n_train_before = len(training_data.dataset)
        n_unlabeled_before = len(unlabeled_data.dataset)
        assert n_train_before == len(
            os.listdir(Path(
                self.yolo_data_path) / "active_learning_train" / "images"))
        assert n_unlabeled_before >= n_samples_to_add, "Not enough data to add"
        if not before_last_training:
            # Initialize a list to store predictions
            unlabeled_predictions = []
            if isinstance(model, torch.nn.Module):
                # Assume you have a dataloader for your unlabeled data
                predictions = []
                self.logger.info("Making RCNN predictions")
                for batch in tqdm(unlabeled_data):
                    # Move the batch to the appropriate device (e.g., GPU)
                    img_batch, _, image_paths = batch
                    img_batch = torch.stack(img_batch)

                    if torch.cuda.is_available():
                        print("GPU is available!")
                        device = "cuda"
                    else:
                        print("GPU is not available.")
                        device = "cpu"

                    batch = img_batch.to(device)

                    # Forward pass through the model
                    model = model.to(device=device)
                    output = RCNN.predict_on_batch(model, batch, image_paths)
                    assert len(batch) == len(output)
                    # Extract the prediction scores (assuming it's a dictionary with "scores" key)
                    prediction = [(pred[0].detach().cpu(), pred[1]) for pred in
                                  output]

                    predictions.extend(prediction)

            elif isinstance(model, YOLOTrainer):
                predictions = model.predict(model_path=yolo_model_path,
                                            images_path=(
                                                    Path(
                                                        self.yolo_data_path) / "active_learning_unlabeled" / "images"),
                                            results_dir=pred_save_dir)
                # for testing only

                # predictions = [
                #     ([[random.random() for _ in range(79)] for _ in
                #       range(5)], image_name) for _, image_name in zip(range(len(os.listdir(Path(
                #         self.yolo_data_path) / "active_learning_unlabeled" / "labels"))), os.listdir(Path(self.yolo_data_path) / "active_learning_unlabeled" / "images"))]
            all_informative_scores = [
                float(max(entropy(pred[0], base=79))) if len(
                    pred[0]) > 0 else 0 for pred
                in predictions]
            # self.logger.info("Unlabeled predictions \n", all_informative_scores)
            # assert len(
            #     all_informative_scores) >= n_samples_to_add, f"Not enough data to add, expected at least {n_samples_to_add} got {len(unlabeled_predictions)}"
            assert len(all_informative_scores) <= len(
                unlabeled_data.dataset), f"We got more predicitions" \
                                         f" than data points {len(all_informative_scores)} > {len(unlabeled_data.dataset)}"
            sorted_indices = np.argsort(all_informative_scores)
            best_indices = sorted_indices[-n_samples_to_add:].tolist()
            # add random indices if too many
            if len(best_indices) < n_samples_to_add:
                random_pool = [el for el in list(range(len(unlabeled_data))) if
                               el not in best_indices]
                random_indices = np.random.choice(random_pool,
                                                  size=n_samples_to_add - len(
                                                      best_indices),
                                                  replace=False).tolist()
                best_indices = best_indices + random_indices
        else:
            assert len(os.listdir(Path(
                self.yolo_data_path) / "active_learning_unlabeled" / "images")) == n_samples_to_add
            best_indices = list(range(len(unlabeled_data.dataset)))
            # mimic predictions
            predictions = [(None, image_name) for image_name in os.listdir(Path(
                self.yolo_data_path) / "active_learning_unlabeled" / "images")]
            assert len(predictions) == n_samples_to_add, len(predictions)
        self.logger.info("Preparing new dataloaders")
        train_data_loader, complement_data_loader = self.get_subsets(
            indices=best_indices, unlabeled_data=unlabeled_data,
            training_data=training_data, batch_size=batch_size)

        os.makedirs(pred_save_dir, exist_ok=True)

        assert len(train_data_loader.dataset) + len(
            complement_data_loader.dataset) == n_unlabeled_before + n_train_before, f"{len(train_data_loader)} + {len(complement_data_loader)} == {n_unlabeled_before} + {n_train}"

        best_images_names = [pred[1] for idx, pred in enumerate(predictions) if
                             idx in best_indices]
        assert len(best_images_names) == n_samples_to_add

        # self.logger.info(best_images_names)
        image_ids = [image_sample["id"] for image_sample in image_data if
                     image_sample["file_name"] in best_images_names]
        # self.logger.info(image_ids)

        return train_data_loader, complement_data_loader, image_ids
