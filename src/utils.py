import json
import logging
import os
import random
import sys
from datetime import datetime
from logging import handlers
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.consts import ODAL_FILEPATH


def setup_logger(output_dir: str):
    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = handlers.RotatingFileHandler(os.path.join(output_dir, "debug.log"), maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)

    return log


def collate_fn(batch):
    return tuple(zip(*batch))


def generate_experiment_name(prefix: str = "experiment") -> str:
    """
    Generate a unique experiment name based on the current date and time.

    Args:
        prefix (str, optional): A prefix for the experiment name. Default is "experiment".

    Returns:
        str: A unique name for the experiment based on the current date and time.
    """
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Create the experiment name
    experiment_name = f"{prefix}_{date_str}"

    return experiment_name


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    # Set the seed for the random number generator
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_add_numbers(proportions: list[int], train_data_loader: DataLoader, complement_data_loader: DataLoader):
    len_train_data = len(train_data_loader.dataset)
    all_data = (len_train_data + len(complement_data_loader.dataset)) / 0.8
    add_data = [int(all_data * add / 100) - len_train_data for add in proportions]
    return [add_data[0]] + [add_data[i] - add_data[i - 1] for i in
                            range(1, len(proportions))] + [0]


def read_full_annotations_file(rcnn_data_path):
    full_annotations_file = Path(rcnn_data_path) / "_annotations.coco.json"

    with open(full_annotations_file, 'r') as file:
        image_data: list[dict] = json.load(file)["images"]

    return image_data


def save_dataloader_to_txt(dataloader, output_file):
    """
    Saves all data from a PyTorch DataLoader to a text file.

    Args:
      dataloader: The PyTorch DataLoader containing the data.
      output_file: The path to the output text file.

    Example usage:
    Assuming you have a DataLoader named 'my_dataloader'
    save_dataloader_to_txt(my_dataloader, 'my_data.txt')
    """

    with open(output_file, 'w') as f:
        for batch in dataloader:
            for data in batch:
                if torch.is_tensor(data):
                    f.write(str(data.tolist()) + '\n')
                else:
                    f.write(str(data) + '\n')


def logg_images_and_labels_from_yolo_dataset(yolo_ds_dir: Path, logger):
    logger.info(f"Printing data numbers from {yolo_ds_dir}")
    for dir_name in os.listdir(yolo_ds_dir):
        n_images = len(os.listdir(yolo_ds_dir / dir_name / "images"))
        n_labels = len(os.listdir(yolo_ds_dir / dir_name / "labels"))
        logger.info(
            f"Dir {dir_name} contains {n_images} images and {n_labels} labels")


def update_odal_config_with_train_subset(subset: str):
    with open(ODAL_FILEPATH, 'r') as f:
        data = yaml.safe_load(f)
    data["train"] = f"{subset}/images"
    with open(ODAL_FILEPATH, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
