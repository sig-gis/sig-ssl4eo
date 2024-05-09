from dataclasses import dataclass
from typing import Literal
from pathlib import Path

@dataclass
class Config:
    imgs_training:str 
    labels_training:str 
    imgs_testing:str 
    labels_testing:str 
    
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
    num_workers:int
    seed:int
    random_subset_frac:float
    checkpoint_key: str = "teacher"
    