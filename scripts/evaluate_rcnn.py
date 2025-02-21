import json
import os

import torch
from jsonargparse import CLI

from src.data.dataset import get_test_data_loaders
from src.experiment_configs import EvaluationConfig
from src.models.rcnn_model import RCNN
from src.utils import setup_logger


def main(experiment_config: EvaluationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(output_dir=experiment_config.output_path)

    test_data_loader = get_test_data_loaders(
        data_dir=experiment_config.data_config.rcnn_data_path,
        batch_size=experiment_config.batch_size)

    results_dir = os.path.join(experiment_config.output_path,
                               "test_evaluations")
    if (not os.path.exists(results_dir)):
        os.mkdir(results_dir)

    for model_path in experiment_config.models_paths:
        results_dir = os.path.join(experiment_config.output_path,
                                   "test_evaluations",
                                   os.path.basename(model_path)[:-4])
        if (not os.path.exists(results_dir)):
            os.mkdir(results_dir)

        model = RCNN.get_model_instance_segmentation(num_classes=11)
        model: RCNN = RCNN.get_model_from_path(model, model_path)

        scores = RCNN.evaluate_on_seperate_dataloader(model=model,
                                                      val_data_loader=test_data_loader,
                                                      device=device)
        logger.info(f"{model_path}: {scores}")
        with open(os.path.join(results_dir, 'scores.json'), 'w') as fp:
            json.dump(scores, fp)


if __name__ == '__main__':
    experiment_config = CLI(EvaluationConfig)
    main(experiment_config)
