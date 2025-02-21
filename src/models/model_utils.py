from src.models.rcnn_model import RCNN
from src.models.yolov5_model import YOLOTrainer


def get_model_class(model_type):
    if model_type == "yolo":
        return YOLOTrainer
    elif model_type == "rcnn":
        return RCNN
    else:
        raise Exception