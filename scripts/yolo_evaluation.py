import json
import os
from pathlib import Path

import tqdm
from jsonargparse import CLI
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data.dataset import get_test_data_loaders
from src.experiment_configs import EvaluationConfig
from src.models.yolov5_model import YOLOTrainer
from src.utils import setup_logger


def get_coco_gt(data_loader):
    all_gt_annotations = []  # list of ground truth annotations in COCO format
    images_info = []  # list of image info dictionaries (id, width, height)
    ann_id = 1  # unique annotation id counter

    # Iterate over the validation DataLoader.
    for imgs, annotations, paths in tqdm.tqdm(data_loader):
        batch_size = len(imgs)
        # Process each image in the current batch.
        for i, (img, path) in enumerate(zip(imgs, paths)):
            # Assume each image is a tensor of shape (C, H, W)
            _, h, w = img.shape
            # Save image info (COCO requires image id, width, height)
            file_name = Path(os.path.basename(path))
            image_stem = int(path.stem) if path.stem.isnumeric() else path.stem
            images_info.append(
                {"id": image_stem, "width": w, "height": h,
                 "file_name": file_name})
            # Process the ground truth annotations for this image.
            ann = annotations[i]
            for box, label in zip(ann["boxes"], ann["labels"]):
                x_min, y_min, x_max, y_max = box.tolist()
                width_box = x_max - x_min
                height_box = y_max - y_min
                all_gt_annotations.append({
                    "id": ann_id,
                    "image_id": image_stem,
                    "category_id": int(label.item()),
                    "bbox": [x_min, y_min, width_box, height_box],
                    "area": width_box * height_box,
                    "iscrowd": 0,
                })
                ann_id += 1
    # Construct a COCO-style ground truth dictionary.
    # Determine all unique category ids from the ground truth.
    category_ids = sorted(
        {ann["category_id"] for ann in all_gt_annotations})
    categories = [{"id": cat_id, "name": str(cat_id)} for cat_id in
                  category_ids]
    coco_gt_dict = {
        "images": images_info,
        "annotations": all_gt_annotations,
        "categories": categories
    }
    import tempfile
    # Write the ground truth dictionary to a temporary JSON file,
    # which is required by the COCO API.
    with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                     suffix='.json') as f:
        json.dump(coco_gt_dict, f)
        gt_file = f.name

    # Create a COCO object for ground truth and load detections.
    return COCO(gt_file)


def main(experiment_config: EvaluationConfig):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(output_dir=experiment_config.output_path)
    test_data_loader = get_test_data_loaders(
        data_dir=experiment_config.data_config.rcnn_data_path,
        batch_size=experiment_config.batch_size)
    coco_gt = get_coco_gt(test_data_loader)

    for model_path in experiment_config.models_paths:
        exp_name = str(os.path.basename(Path(model_path).parents[1]))
        results_dir = os.path.join(experiment_config.output_path,
                                   "test_evaluations",
                                   exp_name)

        os.makedirs(results_dir, exist_ok=True)
        YOLOTrainer.evaluation(model_path,
                               results_dir)
        coco_json_path = os.path.join(results_dir,
                                      "exp", "best_predictions.json")
        coco_dt = coco_gt.loadRes(coco_json_path)

        # Run COCO evaluation.
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract common metrics.
        metrics = {
            "mAP": coco_eval.stats[0],  # x
            # mAP averaged over IoU thresholds 0.50:0.95
            "mAP50": coco_eval.stats[1],  # AP at IoU=0.50 x
            "mAP75": coco_eval.stats[2],  # AP at IoU=0.75
        }

        logger.info(f"{model_path}: {metrics}")

        with open(os.path.join(results_dir, 'scores.json'), 'w') as fp:
            json.dump(metrics, fp)


if __name__ == '__main__':
    experiment_config = CLI(EvaluationConfig)
    main(experiment_config)
