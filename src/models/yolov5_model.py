import ast
import os
from pathlib import Path

import pandas as pd
import torch

from src.consts import ODAL_FILEPATH
from src.utils import generate_experiment_name, \
    update_odal_config_with_train_subset


def read_bbox_from_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    probas = []
    for line in lines:
        class_index, proba, bbox1, bbox2, bbox3, bbox4 = line.split(" ")
        probas.append(float(proba))
    return probas


class YOLOTrainer():

    @staticmethod
    def predict(model_path, images_path, results_dir):
        command = f"python yolov5/detect.py --weights {model_path} " \
                  f"--source {images_path} " \
                  f"--save-csv --save-conf " \
                  f"--conf-thres 0.0001 " \
                  f"--project {results_dir} "
        status = os.system(command)
        if status != 0:
            raise Exception("Command detect failed")
        else:
            all_dirs = os.listdir(results_dir)
            exp_number = max(
                [int(exp[3:]) if len(exp) > 3 else 1 for exp in all_dirs])
            exp_number = exp_number if exp_number > 1 else ""
            labels_dir = Path(
                results_dir) / f"exp{exp_number}" / "predictions.csv"
            # read this data
            df = pd.read_csv(labels_dir,
                             names=["Image Name", "Prediction", "Confidence",
                                    "Probabilities"])
            values = [
                ([(ast.literal_eval(probas)) for probas in image], image_name)
                for image_name, image in
                df.groupby('Image Name')['Probabilities'].apply(
                    list).reset_index().values]
            return values

    @staticmethod
    def evaluation(model_path, results_dir):
        command = f"python yolov5/val.py --weights {model_path} " \
                  f"--data {ODAL_FILEPATH} " \
                  f"--save-json --save-conf " \
                  f"--project {results_dir} " \
                  f"--task test " \
                  f"--half "
        status = os.system(command)
        if status != 0:
            raise Exception("Command failed")

    @staticmethod
    def training_loop(
            num_epochs: int,
            device: str,
            seed: int,
            results_dir: str,
            subset: str,
            patience: int = 5,
            exp_name: str = "test",
            logger=None,
            min_delta: float = 0.01,
            model=None,
            train_data_loader=None,
            val_data_loader=None,
    ) -> None:
        update_odal_config_with_train_subset(subset=subset)
        command = f"python yolov5/train.py " \
                  f"--data {ODAL_FILEPATH} " \
                  f"--weights yolov5s.pt " \
                  f"--img 640 " \
                  f"--epochs {num_epochs} " \
                  f"--device {'0' if device == 'cuda' else device} " \
                  f"--seed {seed} " \
                  f"--project {results_dir} " \
                  f"--name {exp_name} " \
                  f"--patience {patience} " \
                  f"--save-period 1 " \
                  f"--batch-size -1 "
        status = os.system(command)
        if status != 0:
            raise Exception("Command failed")

    @staticmethod
    def get_model_from_path(path: str):
        return YOLOTrainer()

    @staticmethod
    def get_model_instance_segmentation(num_classes: int,
                                        weights: str = 'yolov5s',
                                        device: str = 'cuda') -> torch.nn.Module:
        """
        Load and modify a YOLOv5 model with the specified number of classes.

        Args:
            num_classes (int): The number of classes for the custom dataset.
            weights (str): The version of YOLOv5 to load (e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x').
            device (str): The device to load the model on ('cpu' or 'cuda').

        Returns:
            torch.nn.Module: The modified YOLOv5 model.
        """
        # Load the pretrained YOLOv5 model
        return YOLOTrainer()
        # No need to load the model, it will be loaded during training again
        # model = torch.hub.load('ultralytics/yolov5', weights, pretrained=True).to(device)
        # return model


def test_model():
    train_data_dir = r'/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/export/coco_json/export'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 2 classes; Only target class or background
    # Pomysł: udajmy, że mamy 80 klas
    num_classes = 80  # 80  # lub 2
    num_epochs = 2
    print("Loading the model")
    # model = YOLOTrainer.load_yolov5_model(num_classes=num_classes,
    #                                       weights="yolov5s",
    #                                       device=device)

    # train_data_loader, complement_data_loader, val_data_loader = get_sample_data_loader(data_dir=train_data_dir,
    #                                                                                     batch_size=16)

    # parameters
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    experiment_name = generate_experiment_name(prefix="base_test_yolo")

    results_dir = os.path.join(r'/home/ubuntu/ObjectDetection/experiments',
                               experiment_name)

    print("Training begins")
    YOLOTrainer.training_loop(num_epochs=num_epochs,
                              device=device,
                              seed=1984,
                              results_dir=results_dir,
                              patience=5,
                              min_delta=0.01)


if __name__ == '__main__':
    test_model()
    # from yolov5.train import run
    #
    # config = {}
    # run(**config)

# yolov5.utils.general l#1084
