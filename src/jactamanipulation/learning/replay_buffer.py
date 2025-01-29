# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass

from tensordict import TensorDict
from torchrl.data.replay_buffers import (
    LazyTensorStorage,
    RandomSampler,
    Sampler,
    SliceSampler,
    Storage,
    TensorDictReplayBuffer,
)

from dexterity.planner.planner.parameter_container import ParameterContainer


@dataclass
class ReplayBuffer:
    """ReplayBuffer"""

    params: ParameterContainer
    storage_class: Storage = LazyTensorStorage
    sampler_class: Sampler = RandomSampler

    def __post_init__(self) -> None:
        params = self.params
        self.batch_size = params.learner_batch_size
        self.size = (params.learner_evals + params.learner_max_trajectories) * params.learner_trajectory_length
        self.setup_storage()
        self.setup_sampler()
        self.setup_buffer()

    def setup_storage(self) -> None:
        """Setup storage"""
        self.storage = self.storage_class(self.size, device=self.params.device)

    def setup_sampler(self) -> None:
        """Setup sampler"""
        self.sampler = self.sampler_class()

    def setup_buffer(self) -> None:
        """Setup buffer"""
        self.buffer = TensorDictReplayBuffer(
            storage=self.storage,
            sampler=self.sampler,
            batch_size=self.batch_size,
        )

    def reset(self) -> None:
        """Reset"""
        self.buffer.empty()

    def extend(self, batch: TensorDict) -> None:
        """Extend

        Args:
            batch (TensorDict): Batch to extend with
        """
        self.buffer.extend(batch)

    def sample(self) -> TensorDict:
        """Sample"""
        return self.buffer.sample()

    def __getitem__(self, key: str) -> TensorDict:
        """Get item overload

        Args:
            key (str): key to get Tensor from

        Returns:
            TensorDict: Tensor stored at 'key'
        """
        return self.buffer[key]


@dataclass
class TrajectoryReplayBuffer(ReplayBuffer):
    """TrajectoryReplayBuffer"""

    def setup_sampler(self) -> None:
        """Setup sampler"""
        num_slices_per_sample = self.params.learner_batch_size
        self.sampler = SliceSampler(
            traj_key="episode",
            strict_length=True,
            num_slices=num_slices_per_sample,
        )

    def setup_buffer(self) -> None:
        """Setup buffer"""
        slice_length = self.params.learner_slice_length
        self.batch_size = self.params.learner_batch_size * slice_length
        super().setup_buffer()

    def sample(self) -> TensorDict:
        """Get sample

        Returns:
            TensorDict: Sample
        """
        sample = super().sample()
        return sample.reshape((self.params.learner_batch_size, self.params.learner_slice_length) + sample.shape[1:])
