import os
import shutil
from pathlib import Path

from src.consts import N_TRAIN


def clean_yolo_dataset(yolo_data_path: str | Path, logger):
    yolo_data_path = Path(yolo_data_path)
    AL_TRAIN_DIR = yolo_data_path / "active_learning_train" / "images"
    AL_UNLABELED_DIR = yolo_data_path / "active_learning_unlabeled" / "images"
    # Clean up after previous training and set data for AL
    train_files_before = os.listdir(AL_TRAIN_DIR)
    filenames_in_unlabeled = os.listdir(AL_UNLABELED_DIR)
    for file_path in filenames_in_unlabeled:
        assert file_path not in train_files_before, file_path
    logger.info("Cleaning the AL datasets.")

    # delete all from this directory
    shutil.rmtree(AL_TRAIN_DIR)
    shutil.rmtree(AL_UNLABELED_DIR)

    logger.info("Check if all images are present in AL training")
    # -------------- Copy from INIT to AL_TRAIN --------------
    # 2630
    shutil.copytree(yolo_data_path / "init" / "images",
                    AL_TRAIN_DIR)
    logger.info("Check if all images are present in AL unlabeled")

    # -------------- Copy from DIFF to AL_UNLabeled --------------
    # 18 410
    shutil.copytree(yolo_data_path / "diff" / "images",
                    AL_UNLABELED_DIR)

    assert N_TRAIN == len(os.listdir(AL_UNLABELED_DIR)) + len(
        os.listdir(AL_TRAIN_DIR))

    train_files_before = os.listdir(AL_TRAIN_DIR)
    filenames_in_unlabeled = os.listdir(AL_UNLABELED_DIR)
    for file_path in filenames_in_unlabeled:
        assert file_path not in train_files_before, file_path


def check_for_data_leaks(yolo_ds_dir, logger):
    """
    Checks for data leaks between the all data subsets for active learning.

    Args:
        yolo_ds_dir: Path to the YOLO dataset directory.
        logger: Logger object for logging messages.
    """

    yolo_data_path = Path(yolo_ds_dir)

    def _check_disjoint(ds_1, ds_2):
        ds_1_dir = yolo_data_path / ds_1 / "images"
        ds_2_dir = yolo_data_path / ds_2 / "images"

        train_files = set(os.listdir(ds_1_dir))
        unlabeled_files = set(os.listdir(ds_2_dir))

        # Check for data leaks
        intersection = train_files.intersection(unlabeled_files)

        if intersection:
            logger.error(
                f"Data leaks found between {ds_1} and {ds_2} datasets: {len(intersection)}")

        logger.info(f"No data leaks found between {ds_1} and {ds_2} datasets.")

    _check_disjoint("train", "val")
    _check_disjoint("val", "test")
    _check_disjoint("train", "test")
    _check_disjoint("init", "diff")
    _check_disjoint("active_learning_train", "active_learning_unlabeled")
