# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import glob
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from google.cloud import storage

from dexterity.learning.networks import Actor, Critic
from dexterity.learning.normalizer import Normalizer


def save_tensor(tensor: torch.Tensor, dest_path: Path) -> None:
    """Save tensor to GCP bucket."""
    client = storage.Client(project="proj-dmm")
    bucket = client.get_bucket("project-dmm-main-storage")
    blob = bucket.blob(dest_path)
    with blob.open("wb", ignore_flush=True) as f:
        torch.save(tensor, f)
        print("Tensor uploaded to {}.".format(dest_path))


def load_tensor(src_path: Path) -> torch.Tensor:
    """Load tensor from GCP bucket."""
    client = storage.Client(project="proj-dmm")
    bucket = client.get_bucket("project-dmm-main-storage")
    blob = bucket.blob(src_path)
    with blob.open("rb") as f:
        t = torch.load(f)
        print("Tensor loaded from {}.".format(src_path))
        return t


def save_model(model: Union[Actor, Critic, Normalizer], dest_path: Path, is_local: bool) -> None:
    """Save model either locally or to GCP bucket."""
    if not is_local:
        client = storage.Client(project="proj-dmm")
        bucket = client.get_bucket("project-dmm-main-storage")
        blob = bucket.blob(dest_path)
        f = blob.open("wb", ignore_flush=True)
    else:
        path = Path(dest_path)
        if not path.exists():
            path.parents[0].mkdir(parents=True, exist_ok=True)
            path.touch()
        f = open(path, "wb")
    try:
        model.save(f)
    finally:
        f.close()


def load_model(model: Union[Actor, Critic, Normalizer], src_path: str, is_local: bool) -> None:
    """Load model stored either locally or in a GCP bucket."""
    if not is_local:
        client = storage.Client(project="proj-dmm")
        bucket = client.get_bucket("project-dmm-main-storage")
        path = bucket.blob(src_path)
    else:
        path = Path(src_path)
    with path.open("rb") as f:
        checkpoint = torch.load(f)
        model.load(checkpoint)


def find_latest_model_path(base_path: Path) -> Path:
    """Find directory of the latest model."""
    dirs = glob.glob(base_path.as_posix() + "/*/")  # Lists directories only
    # Sort directories by their timestamp
    latest_dir = sorted(dirs, key=lambda x: datetime.strptime(x.split("/")[-2], "%Y_%m_%d_%H_%M_%S"), reverse=True)[0]
    return Path(latest_dir)
