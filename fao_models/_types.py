from dataclasses import dataclass
from typing import Literal
from pathlib import Path


@dataclass
class ImageryConfig:
    tmp: str | Path
    bands: list[str]
    crops: list[int]
    train_year: int
    predict_year: int


@dataclass
class ProjectConfig:
    eeproject: str
    predict_id: str


@dataclass
class BeamConfig:
    runner: str
    direct_num_workers: int
    direct_running_mode: str


@dataclass
class Config:
    imgs_training: str
    labels_training: str
    imgs_testing: str
    labels_testing: str

    arch: Literal["vit_small"]
    model_root: str | Path
    avgpool_patchtokens: bool
    patch_size: int
    n_last_blocks: int
    lr: float
    batch_size: int
    checkpoints_dir: str | Path
    resume: bool
    epochs: int
    num_workers: int
    seed: int
    random_subset_frac: float
    model_head_root: str | Path
    model_name: str
    imagery_params: ImageryConfig
    project_params: ProjectConfig
    beam_params: BeamConfig
    checkpoint_key: str = "teacher"

    def __post_init__(self):
        self.imagery_params = ImageryConfig(**self.imagery_params)
        self.project_params = ProjectConfig(**self.project_params)
        self.beam_params = BeamConfig(**self.beam_params)
