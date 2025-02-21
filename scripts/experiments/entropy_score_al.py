import os
from pathlib import Path

import torch
from jsonargparse import CLI

from src.al_strategies.MaxEntropyStrategy import MinMaxStrategy
from src.al_strategies.strategy import update_active_training_set_for_yolo
from src.consts import NUM_CLASSES
from src.data.dataset import get_base_data_loaders
from src.data.yolo_utils import clean_yolo_dataset
from src.experiment_configs import ExperimentConfig
from src.models.model_utils import get_model_class
from src.utils import get_add_numbers, generate_experiment_name, setup_logger, \
    read_full_annotations_file


def main(experiment_config: ExperimentConfig):
    ModelClass = get_model_class(experiment_config.model_config.model_type)

    image_data = read_full_annotations_file(
        experiment_config.data_config.rcnn_data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name = generate_experiment_name(
        prefix=experiment_config.exp_name)
    results_dir = os.path.join(experiment_config.output_path, experiment_name)
    os.mkdir(results_dir)
    logger = setup_logger(output_dir=results_dir)

    yolo_data_path = Path(experiment_config.data_config.yolo_data_path)

    if experiment_config.model_config.model_type == "yolo":
        clean_yolo_dataset(yolo_data_path, logger)

    train_data_loader, _, complement_data_loader, val_data_loader = get_base_data_loaders(
        data_dir=experiment_config.data_config.rcnn_data_path,
        batch_size=experiment_config.batch_size)

    strategy = MinMaxStrategy(
        yolo_data_path=yolo_data_path,
        logger=logger)

    add_numbers = get_add_numbers(proportions=[15, 20, 25, 30, 40, 60, 80],
                                  train_data_loader=train_data_loader,
                                  complement_data_loader=complement_data_loader)

    # Pytanie - w każdym kroku trenuję model 'od początku' (od pretrenowanego modelu z neta)

    for i, n_samples_to_add in enumerate(add_numbers):
        before_last_training = i == len(add_numbers) - 2

        logger.info(f"Begin AL {i} iteration")
        n_labeled_data = len(train_data_loader.dataset)
        n_unlabeled_data = len(complement_data_loader.dataset)

        assert n_unlabeled_data == len(
            os.listdir(Path(
                yolo_data_path) / "active_learning_unlabeled" / "images"))
        assert n_labeled_data == len(
            os.listdir(Path(
                yolo_data_path) / "active_learning_train" / "images"))

        logger.info(
            f"Using {n_labeled_data} training samples"
            f" and {n_unlabeled_data} unlabeled samples")

        run_name = f"run_{i}"

        if i == 0:
            model_path = experiment_config.saved_models.first_rcnn_model_path if experiment_config.model_config.model_type == "rcnn" else experiment_config.saved_models.first_yolo_model_path
            yolo_model_path = model_path
            model: torch.nn.Module = ModelClass.get_model_from_path(
                path=model_path)
        elif i == len(add_numbers) - 1:
            model_path = experiment_config.saved_models.last_rcnn_model_path if experiment_config.model_config.model_type == "rcnn" else experiment_config.saved_models.last_yolo_model_path
            model: torch.nn.Module = ModelClass.get_model_from_path(
                path=model_path)
        else:
            yolo_model_path = Path(
                results_dir) / run_name / "weights" / "best.pt" if experiment_config.model_config.model_type == "yolo" else None
            model: torch.nn.Module = ModelClass.get_model_instance_segmentation(
                num_classes=NUM_CLASSES)

            ModelClass.training_loop(model=model,
                                     num_epochs=experiment_config.epochs,
                                     train_data_loader=train_data_loader,
                                     val_data_loader=val_data_loader,
                                     subset="active_learning_train",
                                     device=device,
                                     seed=experiment_config.seed,
                                     results_dir=results_dir,
                                     exp_name=run_name,
                                     logger=logger)
            if experiment_config.model_config.model_type == "rcnn":
                torch.save(model.state_dict(),
                           os.path.join(results_dir,
                                        f"rcnn_model_weights_{len(train_data_loader)}.pth"))

        if n_samples_to_add > 0:
            logger.info(f"Adding {n_samples_to_add} samples")
            train_data_loader, complement_data_loader, image_ids = strategy.get_next_dataset(
                model=model,
                training_data=train_data_loader,
                unlabeled_data=complement_data_loader,
                batch_size=experiment_config.batch_size,
                n_samples_to_add=n_samples_to_add,
                yolo_model_path=yolo_model_path,
                pred_save_dir=Path(results_dir) / run_name / "validation",
                before_last_training=before_last_training,
                image_data=image_data
            )
            assert len(
                image_ids) == n_samples_to_add, f"Expected {n_samples_to_add} files but got {len(image_ids)}"
            if experiment_config.model_config.model_type == "yolo":
                update_active_training_set_for_yolo(image_data=image_data,
                                                    image_ids=image_ids,
                                                    data_path=yolo_data_path,
                                                    logger=logger)
                n_unlabeled_data_after = len(os.listdir(
                    Path(
                        yolo_data_path) / "active_learning_unlabeled" / "images"))
                n_labeled_data_after = len(os.listdir(
                    Path(
                        yolo_data_path) / "active_learning_train" / "images"))
                assert n_unlabeled_data_after == n_unlabeled_data - n_samples_to_add
                assert n_labeled_data_after == n_labeled_data + n_samples_to_add
                assert len(complement_data_loader.dataset) == len(
                    os.listdir(Path(
                        yolo_data_path) / "active_learning_unlabeled" / "images"))
                assert len(train_data_loader.dataset) == len(
                    os.listdir(Path(
                        yolo_data_path) / "active_learning_train" / "images"))


if __name__ == '__main__':
    experiment_config = CLI(ExperimentConfig)
    main(experiment_config)
