# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import random
import unittest

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import RandomSampler, SliceSampler

from jactamanipulation.learning.replay_buffer import ReplayBuffer, TrajectoryReplayBuffer
from jactamanipulation.planner.planner.parameter_container import ParameterContainer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.params = ParameterContainer()
        self.params.learner_evals = 10
        self.params.learner_max_trajectories = 20
        self.params.learner_trajectory_length = 5
        self.params.learner_batch_size = 32
        self.params.device = "cpu"
        self.setup_buffer()

    def setup_buffer(self) -> None:
        self.replay_buffer = ReplayBuffer(self.params)

    def test_initialization(self) -> None:
        assert self.replay_buffer.size == 150  # (10 + 20) * 5
        assert isinstance(self.replay_buffer.sampler, RandomSampler)

    def dummy_experience(self) -> TensorDict:
        return TensorDict(
            {
                "states": torch.randn((1, 27), device=self.params.device),
                "actions": torch.randn((1, 7), device=self.params.device),
                "rewards": torch.randn((1, 1), device=self.params.device),
                "done": torch.randint(0, 2, (1, 1), device=self.params.device).bool(),
                "sensors": torch.randint(0, 255, (1, 3, 128, 128), dtype=torch.uint8, device=self.params.device),
            },
            batch_size=1,
        )

    def test_extend(self) -> None:
        experience = self.dummy_experience()
        self.replay_buffer.extend(experience)
        assert len(self.replay_buffer.buffer) == 1

    def test_sample(self) -> None:
        for _ in range(10):  # Add 10 experiences
            self.replay_buffer.extend(self.dummy_experience())

        sample = self.replay_buffer.sample()
        assert isinstance(sample, TensorDict)
        assert sample["states"].shape == (32, 27)  # batch_size = 32
        assert sample["actions"].shape == (32, 7)
        assert sample["rewards"].shape == (32, 1)
        assert sample["done"].shape == (32, 1)
        assert sample["sensors"].shape == (32, 3, 128, 128)

    def test_getitem(self) -> None:
        for _ in range(10):  # Add 10 experiences
            self.replay_buffer.extend(self.dummy_experience())

        states = self.replay_buffer["states"]
        assert states.shape[0] == 10  # number of experiences
        assert states.shape[1] == 27  # state dimension

    def test_reset(self) -> None:
        for _ in range(10):  # Add 10 experiences
            self.replay_buffer.extend(self.dummy_experience())

        self.replay_buffer.reset()
        assert len(self.replay_buffer.buffer) == 0


class TestTrajectoryReplayBuffer(TestReplayBuffer):
    def setup_buffer(self) -> None:
        self.params.learner_slice_length = 8
        self.replay_buffer = TrajectoryReplayBuffer(self.params)

    def test_initialization(self) -> None:
        print(self.replay_buffer.sampler)
        assert isinstance(self.replay_buffer.sampler, SliceSampler)
        assert self.replay_buffer.batch_size == self.params.learner_batch_size * self.params.learner_slice_length

    def test_sampler_configuration(self) -> None:
        sampler = self.replay_buffer.sampler
        assert sampler.traj_key == "episode"
        assert sampler.num_slices == self.params.learner_batch_size

    def create_sequence_experience(self, sequence_length: int) -> TensorDict:
        return TensorDict(
            {
                "states": torch.randn((sequence_length, 27), device=self.params.device),
                "actions": torch.randn((sequence_length, 7), device=self.params.device),
                "rewards": torch.randn((sequence_length, 1), device=self.params.device),
                "done": torch.randint(0, 2, (sequence_length, 1), device=self.params.device).bool(),
                "sensors": torch.randint(
                    0, 255, (sequence_length, 3, 128, 128), dtype=torch.uint8, device=self.params.device
                ),
                "episode": torch.full((sequence_length,), 0, device=self.params.device),
            },
            batch_size=[sequence_length],
        )

    def test_extend(self) -> None:
        sequence_length = 20
        experience = self.create_sequence_experience(sequence_length)
        self.replay_buffer.extend(experience)
        assert len(self.replay_buffer.buffer) == sequence_length

    def test_sample(self) -> None:
        # Add multiple sequences
        for i in range(5):
            seq_length = random.randint(20, 30)
            experience = self.create_sequence_experience(seq_length)
            experience["episode"] = torch.full((seq_length,), i, device=self.params.device)
            self.replay_buffer.extend(experience)

        sample = self.replay_buffer.sample()
        assert isinstance(sample, TensorDict)

        expected_batch_size = self.params.learner_batch_size
        expected_slice_length = self.params.learner_slice_length
        assert sample["states"].shape == (expected_batch_size, expected_slice_length, 27)
        assert sample["actions"].shape == (expected_batch_size, expected_slice_length, 7)
        assert sample["rewards"].shape == (expected_batch_size, expected_slice_length, 1)
        assert sample["done"].shape == (expected_batch_size, expected_slice_length, 1)
        assert sample["sensors"].shape == (expected_batch_size, expected_slice_length, 3, 128, 128)
        assert sample["episode"].shape == (expected_batch_size, expected_slice_length)

        # Check that each slice contains consecutive timesteps from the same episode
        episodes = sample["episode"]
        # All of the timesteps in each slice should have the same episode number
        assert torch.all(episodes == episodes[:, 0].unsqueeze(1))

    def test_reset(self) -> None:
        for i in range(2):
            experience = self.create_sequence_experience(10)
            experience["episode"] = torch.full((10, 1), i, device=self.params.device)
            self.replay_buffer.extend(experience)

        self.replay_buffer.reset()
        assert len(self.replay_buffer.buffer) == 0
