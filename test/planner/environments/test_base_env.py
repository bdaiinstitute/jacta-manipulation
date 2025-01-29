# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import unittest

import torch

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.environments import DexterityEnv
from jactamanipulation.planner.planner.parameter_container import ParameterContainer


class TestDexterityEnv(unittest.TestCase):
    device: torch.device
    params: ParameterContainer
    plant: MujocoPlant

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.params = ParameterContainer("dexterity/examples/planner/config/planar_hand.yml")
        cls.params.success_progress_threshold = 0.05
        cls.params.learner_trajectory_length = 50
        cls.params.num_envs = 2

    def setUp(self) -> None:
        self.env = DexterityEnv(params=self.params)

    def test_initialization(self) -> None:
        self.assertEqual(self.env.num_envs, self.params.num_envs)
        self.assertEqual(self.env.params, self.params)
        self.assertIsNotNone(self.env.action_processor)
        self.assertIsNotNone(self.env.plant)
        self.assertEqual(self.env.episode_length, self.params.learner_trajectory_length)
        self.assertTrue(torch.all(self.env.action_magnitude == self.params.action_range * self.params.action_time_step))

    def test_observation_space(self) -> None:
        expected_shape = (self.env.plant.state_dimension * 2,)
        self.assertEqual(self.env.observation_space.shape, expected_shape)

    def test_action_space(self) -> None:
        expected_shape = (self.env.plant.action_dimension,)
        self.assertEqual(self.env.action_space.shape, expected_shape)
        self.assertTrue(
            torch.all(self.env.action_space.low == -self.params.action_range * self.params.action_time_step)
        )
        self.assertTrue(
            torch.all(self.env.action_space.high == self.params.action_range * self.params.action_time_step)
        )

    def test_uniform_random_action(self) -> None:
        for _ in range(5):
            action = self.env.uniform_random_action()
            self.assertEqual(action.shape, (self.params.num_envs, self.env.plant.action_dimension))
            self.assertTrue(torch.all(action >= -self.params.action_range * self.params.action_time_step))
            self.assertTrue(torch.all(action <= self.params.action_range * self.params.action_time_step))

    def test_reset(self) -> None:
        obs, info = self.env.reset()

        self.assertEqual(obs.shape, (self.params.num_envs, self.env.plant.state_dimension * 2))
        self.assertIn("done", info)
        self.assertIn("timestep", info)
        self.assertIn("is_success", info)
        self.assertIn("is_failure", info)
        self.assertIn("progress", info)
        self.assertIn("scaled_distance_to_goal", info)

    def test_step(self) -> None:
        self.env.reset()
        action = self.env.uniform_random_action()
        obs, rewards, termination, truncation, info = self.env.step(action)

        self.assertEqual(obs.shape, (self.params.num_envs, self.env.plant.state_dimension * 2))
        self.assertEqual(rewards.shape, (self.params.num_envs,))
        self.assertEqual(termination.shape, (self.params.num_envs,))
        self.assertEqual(truncation.shape, (self.params.num_envs,))
        self.assertIn("done", info)
        self.assertIn("is_success", info)
        self.assertIn("is_failure", info)
        self.assertIn("progress", info)
        self.assertIn("scaled_distance_to_goal", info)

    def test_get_metrics(self) -> None:
        self.env.reset()
        metrics = self.env.get_metrics()

        self.assertIn("scaled_distance_to_goal", metrics)
        self.assertIn("progress", metrics)
        self.assertEqual(metrics["scaled_distance_to_goal"].shape, (self.params.num_envs,))
        self.assertEqual(metrics["progress"].shape, (self.params.num_envs,))

    def test_multiple_steps(self) -> None:
        self.env.reset()
        for _ in range(self.params.learner_trajectory_length * 2):
            action = self.env.uniform_random_action()
            obs, rewards, termination, truncation, info = self.env.step(action)

            self.assertEqual(obs.shape, (self.params.num_envs, self.env.plant.state_dimension * 2))
            self.assertEqual(rewards.shape, (self.params.num_envs,))
            self.assertEqual(termination.shape, (self.params.num_envs,))
            self.assertEqual(truncation.shape, (self.params.num_envs,))
            self.assertIn("done", info)
            self.assertLessEqual(info["timestep"][0], self.params.learner_trajectory_length - 1)

            if info["timestep"][0] == self.params.learner_trajectory_length - 1:
                self.assertTrue(info["done"][0])
                # Very likely that random actions will not reach the goal and truncate
                self.assertTrue(truncation[0])
                self.assertFalse(termination[0])

    def test_rendering(self) -> None:
        self.env.reset()
        action = self.env.uniform_random_action()
        self.env.step(action)

        self.assertIsNotNone(self.env.state_trajectory)

        # This test doesn't actually render, but checks if the state can be extracted
        for state in self.env.state_trajectory[0].cpu().numpy():
            self.assertEqual(state.shape, (self.env.plant.state_dimension,))
