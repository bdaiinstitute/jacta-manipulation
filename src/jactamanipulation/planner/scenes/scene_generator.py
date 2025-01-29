# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


if __name__ == "__main__":
    # Load the scenes
    from dexterity.jacta_planner.scenes import scene_registry

    # Generate all scenes
    scene_registry.generate_all()
