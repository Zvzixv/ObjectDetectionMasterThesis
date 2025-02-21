"""
Open annotations in CoCo format and convert to Yolo dataset
"""
import os
import shutil
from pathlib import Path

from src.data.conversion_utils import make_dirs
from src.data.json2yolo import convert_coco_json
from src.data.yolo_utils import check_for_data_leaks
from src.utils import setup_logger, logg_images_and_labels_from_yolo_dataset

# argparse
# Gdyby nie bylo folderu unlabeled to uruchomic ten skrypt
if __name__ == "__main__":
    # --------------------- SETUP THIS -----------------------------
    input_json_dir = r"/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/export/coco_json"
    new_yolo_dataset_name = "yolo_format_data_test_13"
    logger_dir = r"/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/temp"
    # --------------------------------------------------------------

    Path(logger_dir).mkdir(exist_ok=True)
    logger = setup_logger(logger_dir)

    yolo_ds_dir = Path(input_json_dir) / new_yolo_dataset_name
    assert not os.path.exists(
        yolo_ds_dir), "Create a new dataset, cannot update an existing one"
    logger.info("This script takes more than 24 minutes to run...")
    logger.info("Data conversion, should take around 12 minutes.")
    convert_coco_json(
        json_dir=input_json_dir,  # directory with *.json
        save_dir=yolo_ds_dir,  # output directory
        use_segments=False,
        cls91to80=True,
    )

    # Create active learning train ds
    make_dirs(yolo_ds_dir / "active_learning_train")

    shutil.copytree(yolo_ds_dir / "init" / "labels",
                    yolo_ds_dir / "active_learning_train" / "labels",
                    dirs_exist_ok=True)
    logger.info("Creating a copy of init images.")
    # copy images
    make_dirs(yolo_ds_dir / "active_learning_unlabeled")
    # # Create the diff folder
    shutil.copytree(yolo_ds_dir / "train" / "labels",
                    yolo_ds_dir / "diff" / "labels",
                    dirs_exist_ok=True)
    shutil.copytree(yolo_ds_dir / "train" / "images",
                    yolo_ds_dir / "diff" / "images",
                    dirs_exist_ok=True)
    for file_name in os.listdir(yolo_ds_dir / "init" / "images"):
        os.remove(yolo_ds_dir / "diff" / "images" / file_name)

    shutil.copytree(yolo_ds_dir / "diff" / "labels",
                    yolo_ds_dir / "active_learning_unlabeled" / "labels",
                    dirs_exist_ok=True)

    logg_images_and_labels_from_yolo_dataset(yolo_ds_dir, logger=logger)
    check_for_data_leaks(yolo_ds_dir, logger=logger)
