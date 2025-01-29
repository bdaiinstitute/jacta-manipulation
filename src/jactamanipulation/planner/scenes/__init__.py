# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from dexterity.planner.scenes.scene_composer import SpotRobot
from dexterity.planner.scenes.scene_registry import scene_registry

# Define what should be exported from this module
__all__ = ["scene_registry", "spot_sensors"]

# TODO: add manually created scenes

spot_sensors = [
    dict(type="framepos", name="sensor_body", objname="spot/base/site_body", refname="object/site_object"),
    dict(type="frameyaxis", name="sensor_object_y_axis", objname="object/site_object"),
    dict(type="framepos", name="sensor_fngr_obj", objname="spot/arm/site_arm_link_fngr", refname="object/site_object"),
    dict(type="framepos", name="trace_body", objname="spot/base/site_body"),
    dict(type="framepos", name="trace_gripper", objname="spot/arm/site_arm_link_fngr"),
]

spot_contacts = [
    dict(body1="spot/arm/arm_link_sh0", body2="spot/arm/arm_link_el1"),
    dict(body1="spot/arm/arm_link_sh1", body2="spot/arm/arm_link_el0"),
    dict(body1="spot/arm/arm_link_sh1", body2="spot/arm/arm_link_el1"),
    dict(body1="spot", body2="spot/legs/front_left_upper_leg"),
    dict(body1="spot", body2="spot/legs/front_right_upper_leg"),
    dict(body1="spot", body2="spot/legs/rear_left_upper_leg"),
    dict(body1="spot", body2="spot/legs/rear_right_upper_leg"),
]

scene_registry.add(
    name="spot_tire",
    elements={
        "ground": "common/ground",
        "spot": SpotRobot(),
        "object": "objects/tire_rim",
    },
    sensors=spot_sensors,
    contacts=spot_contacts,
)

scene_registry.add(
    name="spot_box",
    elements={
        "ground": "common/ground",
        "spot": SpotRobot(),
        "object": "objects/box",
    },
    sensors=spot_sensors,
    contacts=spot_contacts,
)

scene_registry.add(
    name="spot_stool",
    elements={
        "ground": "common/ground",
        "spot": SpotRobot(),
        "object": "objects/stool",
    },
    sensors=spot_sensors,
    contacts=spot_contacts,
)
