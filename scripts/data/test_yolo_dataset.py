"This scripts counts data and makes checks for yolo ds"
from pathlib import Path

# Gdyby nie bylo folderu unlabeled to uruchomic ten skrypt
from src.data.yolo_utils import check_for_data_leaks
from src.utils import logg_images_and_labels_from_yolo_dataset, setup_logger

if __name__ == "__main__":
    logger_dir = r"/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/temp"
    # --------------------------------------------------------------

    Path(logger_dir).mkdir(exist_ok=True)
    logger = setup_logger(logger_dir)
    yolo_ds_dir = Path(
        r"/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/export/coco_json/yolo_format_data_test_14")
    logg_images_and_labels_from_yolo_dataset(yolo_ds_dir, logger=logger)

    check_for_data_leaks(yolo_ds_dir, logger=logger)
