from dataclasses import dataclass



@dataclass
class DataConfig:
    rcnn_data_path: str
    yolo_data_path: str


@dataclass
class ModelConfig:
    model_type: str


@dataclass
class SavedModels:
    first_rcnn_model_path: str = None
    last_rcnn_model_path: str = None
    first_yolo_model_path: str = None
    last_yolo_model_path: str = None


@dataclass
class EvaluationConfig:
    data_config: DataConfig
    model_config: ModelConfig
    models_paths: list[str]
    output_path: str
    batch_size: int = 16


@dataclass
class ExperimentConfig:
    data_config: DataConfig
    model_config: ModelConfig
    saved_models: SavedModels
    exp_name: str
    output_path: str
    batch_size: int = 16
    epochs: int = 1
    seed: int = 1984
