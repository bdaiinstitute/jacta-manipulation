# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from jactamanipulation.tasks.acrobot import Acrobot, AcrobotConfig
from jactamanipulation.tasks.cartpole import Cartpole, CartpoleConfig
from jactamanipulation.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from jactamanipulation.tasks.spot.spot_box import SpotBox, SpotBoxConfig
from jactamanipulation.tasks.spot.spot_hand_navigation import SpotHandNavigation, SpotHandNavigationConfig
from jactamanipulation.tasks.spot.spot_navigation import SpotNavigation, SpotNavigationConfig
from jactamanipulation.tasks.spot.spot_tire import SpotTire, SpotTireConfig
from jactamanipulation.tasks.spot_position_control import SpotPositionControl, SpotPositionControlConfig
from jactamanipulation.tasks.task import Task, TaskConfig

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


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    _registered_tasks[name] = (task_type, task_config_type)
