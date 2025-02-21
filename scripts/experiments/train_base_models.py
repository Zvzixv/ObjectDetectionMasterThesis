import os

import torch
from jsonargparse import CLI

from src.consts import NUM_CLASSES
from src.data.dataset import get_base_data_loaders
from src.experiment_configs import ExperimentConfig
from src.models.model_utils import get_model_class
from src.utils import generate_experiment_name, setup_logger


def main(experiment_config: ExperimentConfig):
    ModelClass = get_model_class(experiment_config.model_config.model_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_name = generate_experiment_name(
        prefix="base_models_" + experiment_config.exp_name)
    results_dir = os.path.join(experiment_config.output_path, experiment_name)
    os.mkdir(results_dir)
    logger = setup_logger(output_dir=results_dir)

    init_data_loader, train_data_loader, _, val_data_loader = get_base_data_loaders(
        data_dir=experiment_config.data_config.rcnn_data_path,
        batch_size=experiment_config.batch_size)

    n_train_data = len(train_data_loader.dataset)
    n_init_data = len(init_data_loader.dataset)

    logger.info(
        f"Using {n_train_data} training samples for last training"
        f" and {n_init_data} init samples for first")
    init_model: torch.nn.Module = ModelClass.get_model_instance_segmentation(
        num_classes=NUM_CLASSES)

    ModelClass.training_loop(model=init_model,
                             num_epochs=experiment_config.epochs,
                             train_data_loader=init_data_loader,
                             val_data_loader=val_data_loader,
                             device=device,
                             seed=experiment_config.seed,
                             results_dir=results_dir,
                             exp_name="init_base",
                             subset="init",
                             logger=logger)

    if experiment_config.model_config.model_type == "rcnn":
        torch.save(init_model.state_dict(),
                   os.path.join(results_dir,
                                f"rcnn_init_model_weights_{len(train_data_loader)}.pth"))

    full_model: torch.nn.Module = ModelClass.get_model_instance_segmentation(
        num_classes=NUM_CLASSES)
    ModelClass.training_loop(model=full_model,
                             num_epochs=experiment_config.epochs,
                             train_data_loader=train_data_loader,
                             val_data_loader=val_data_loader,
                             device=device,
                             seed=experiment_config.seed,
                             results_dir=results_dir,
                             subset="train",
                             exp_name="full_base",
                             logger=logger)

    if experiment_config.model_config.model_type == "rcnn":
        torch.save(full_model.state_dict(),
                   os.path.join(results_dir,
                                f"rcnn_full_model_weights_{len(train_data_loader)}.pth"))


if __name__ == '__main__':
    experiment_config = CLI(ExperimentConfig)
    main(experiment_config)
