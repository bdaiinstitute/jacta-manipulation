# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from jacta.tasks.acrobot import Acrobot, AcrobotConfig
from jacta.tasks.cartpole import Cartpole, CartpoleConfig
from jacta.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from jacta.tasks.spot.spot_box import SpotBox, SpotBoxConfig
from jacta.tasks.spot.spot_hand_navigation import (
    SpotHandNavigation,
    SpotHandNavigationConfig,
)
from jacta.tasks.spot.spot_navigation import SpotNavigation, SpotNavigationConfig
from jacta.tasks.spot.spot_tire import SpotTire, SpotTireConfig
from jacta.tasks.spot_position_control import (
    SpotPositionControl,
    SpotPositionControlConfig,
)
from jacta.tasks.task import Task, TaskConfig

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    "acrobot": (Acrobot, AcrobotConfig),
    "cylinder_push": (CylinderPush, CylinderPushConfig),
    "cartpole": (Cartpole, CartpoleConfig),
    "spot_box": (SpotBox, SpotBoxConfig),
    "spot_hand_navigation": (SpotHandNavigation, SpotHandNavigationConfig),
    "spot_navigation": (SpotNavigation, SpotNavigationConfig),
    "spot_position_control": (SpotPositionControl, SpotPositionControlConfig),
    "spot_tire": (SpotTire, SpotTireConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    return _registered_tasks


def register_task(
    name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]
) -> None:
    _registered_tasks[name] = (task_type, task_config_type)
