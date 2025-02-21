import numpy as np
from scipy.stats import entropy

from src.models.yolov5_model import YOLOTrainer

if __name__ == '__main__':
    # model_path = r"C:\Users\skrzy\Documents\best.pt"
    # images_path = r"C:\Users\skrzy\Documents\SelfDrivingCarCOCO\coco_json\yolo_format_data_test_7\val\images"
    # results_dir = r"C:\Users\skrzy\Documents\yolov5_test_detect"

    # Zuzia
    model_path = r"/home/ubuntu/ObjectDetection/experiments/yolo_test_20241221_122143/run_7/weights/best.pt"
    images_path = r"/home/ubuntu/test_photos"
    results_dir = r"/home/ubuntu/yolov5_test_detect"

    unlabeled_predictions = YOLOTrainer.predict(model_path, images_path,
                                                results_dir)
    print(len(unlabeled_predictions))
    print(len(unlabeled_predictions[2][0]))
    print(entropy(unlabeled_predictions[0], base=79))
    print(len(entropy(unlabeled_predictions[0], base=79)))
    all_informative_scores = [
        max(entropy(pred, base=79)) if len(pred) > 0 else 0 for pred
        in unlabeled_predictions]
    print("Unlabeled predictions \n", all_informative_scores)
    # assert len(
    #     all_informative_scores) >= n_samples_to_add, f"Not enough data to add, expected at least {n_samples_to_add} got {len(unlabeled_predictions)}"
    print(len(all_informative_scores))

    sorted_indices = np.argsort(all_informative_scores)
    highest_indices = sorted_indices[-2:].tolist()
    print(highest_indices)
