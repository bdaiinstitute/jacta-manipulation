# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from multiprocessing.managers import SyncManager


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
        self.spot_hardware_process_running = manager.Event()
