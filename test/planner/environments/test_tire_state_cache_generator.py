# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import unittest

from dexterity.planner.environments.tire_state_cache_generator import TireGeneratorConfig, TireStateCacheGenerator


class TestFloatingSpotTireCacheGenerator(unittest.TestCase):

    def setUp(self) -> None:
        task_name = "spot_floating_tire"
        config = TireGeneratorConfig(
            task=task_name,
            cache_size=10,
            enable_rendering=False,
        )
        self.generator = TireStateCacheGenerator(task_name=task_name, config=config)

    def test_generate_cache(self) -> None:
        cache = self.generator.generate_cache()
        self.assertEqual(cache.shape[0], self.generator.config.cache_size)
        self.assertEqual(cache.shape[1], self.generator.plant.model.nq + self.generator.plant.model.nv)

    def test_generate_single_state(self) -> None:
        state = self.generator._generate_single_state()
        self.assertEqual(state.shape[0], self.generator.plant.model.nq + self.generator.plant.model.nv)

    def test_randomize_object_pose(self) -> None:
        pose = self.generator._randomize_object_pose()
        self.assertEqual(pose.shape[0], 7)

    def test_apply_velocity_perturbation(self) -> None:
        state = self.generator._generate_single_state()
        perturbed_state = self.generator._apply_velocity_perturbation(state)
        self.assertEqual(perturbed_state.shape[0], state.shape[0])
