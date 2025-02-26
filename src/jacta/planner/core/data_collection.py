# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import glob
from datetime import datetime
from pathlib import Path
from typing import Union

import torch

from jacta.learning.networks import Actor, Critic
from jacta.learning.normalizer import Normalizer


def save_model(model: Union[Actor, Critic, Normalizer], dest_path: str) -> None:
    path = Path(dest_path)
    if not path.exists():
        path.parents[0].mkdir(parents=True, exist_ok=True)
        path.touch()
    f = open(path, "wb")
    try:
        model.save(f)
    finally:
        f.close()


def load_model(model: Union[Actor, Critic, Normalizer], src_path: str) -> None:
    path = Path(src_path)
    with path.open("rb") as f:
        checkpoint = torch.load(f)
        model.load(checkpoint)


def find_latest_model_path(base_path: str) -> str:
    dirs = glob.glob(base_path + "*/")  # Lists directories only
    # Sort directories by their timestamp
    latest_dir = sorted(
        dirs,
        key=lambda x: datetime.strptime(x.split("/")[-2], "%Y_%m_%d_%H_%M_%S"),
        reverse=True,
    )[0]
    return latest_dir
