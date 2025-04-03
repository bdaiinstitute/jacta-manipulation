# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass
from multiprocessing.managers import SyncManager


@dataclass
class ControlBufferKeys:
    """Keys of the control buffer in the IO context."""

    all_traces_rollout_size: str = "all_traces_rollout_size"
    spline: str = "spline"
    traces: str = "traces"


@dataclass
class StateBufferKeys:
    """Keys of the state buffer in the IO context."""

    additional_info: str = "additional_info"
    last_policy_output: str = "last_policy_output"
    qpos: str = "qpos"
    qvel: str = "qvel"
    time: str = "time"
    xpos: str = "xpos"
    xquat: str = "xquat"


@dataclass
class HardwareStateBufferKeys(StateBufferKeys):
    """Keys of the hardware state buffer in the IO context."""

    spot_hardware_state: str = "spot_hardware_state"


class IOContext:
    """Container for all multiprocessing I/O objects needed for Viser app."""

    def __init__(self, manager: SyncManager):
        # Store manager to enable recreation of buffers.
        self.manager = manager

        # Shared state and control buffers
        self.state_buffer = manager.dict()
        self.control_buffer = manager.dict()

        # Locks for synchronization
        self.state_lock = manager.Lock()
        self.control_lock = manager.Lock()

        # Configuration dictionaries and their locks
        self.control_config_dict = manager.dict()
        self.reward_config_dict = manager.dict()
        self.control_config_lock = manager.Lock()
        self.reward_config_lock = manager.Lock()

        # Events for signaling updates
        self.control_config_updated_event = manager.Event()
        self.reward_config_updated_event = manager.Event()
        self.task_updated_event = manager.Event()

        # Events for signaling controller + sim lifecycling.
        self.controller_running = manager.Event()
        self.simulation_running = manager.Event()
        self.simulation_reset_event = manager.Event()

        # Events for signaling when to flip the profiling on and off
        self.profiler_updated_event = manager.Event()

        # Profiling stats
        self.profiling_lock = manager.Lock()
        self.profiling_stats_dict = manager.dict()

        # Visualizer lock
        self.visualization_lock = manager.Lock()


class SimulationIOContext(IOContext):
    """Container for all multiprocessing I/O objects needed for Viser app."""

    def __init__(self, manager: SyncManager):
        super(SimulationIOContext, self).__init__(manager)


class HardwareIOContext(IOContext):
    """Container for all multiprocessing I/O objects needed for Viser app."""

    def __init__(self, manager: SyncManager):
        super(HardwareIOContext, self).__init__(manager)

        # Events for signaling hardware components
        self.mocap_process_running = manager.Event()
        self.hardware_process_running = manager.Event()
