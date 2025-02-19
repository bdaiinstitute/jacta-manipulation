# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import cProfile
import os
import pstats
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class ViserProfilerConfig:
    """Configuration file for the Viser profiler.

    Tracks the fields that the config is looking at.
    """

    tracked_fields: Dict = field(
        default_factory=lambda: {
            "control_loop": "update_action",
        }
    )


class ViserProfiler:
    """Profiler that generates profiling and speed statistics of Viser."""

    def __init__(self, logfile: Optional[str] = None, record: bool = False) -> None:
        self.profiler = cProfile.Profile()
        self.logfile = logfile
        if self.logfile is not None:
            os.makedirs(os.path.dirname(self.logfile), exist_ok=True)
        self.recording = record
        self.config = ViserProfilerConfig()
        self.agg_stats: dict = {}
        for key in self.config.tracked_fields.keys():
            self.agg_stats[key] = None

    def benchmark_function(
        self, input_function: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Call a function with a benchmarked process"""
        self.profiler.enable()
        outputs = input_function(*args, **kwargs)
        self.profiler.create_stats()
        if self.logfile is None:
            self.profiler.print_stats()
        else:
            self.profiler.dump_stats(self.logfile)
        return outputs

    def benchmark_wrapper(self, input_function: Callable) -> Any:
        """Wrapper to convert a function into a profiled one"""

        def perf_wrapper(*args: Any, **kwargs: Any) -> Any:
            self.profiler.enable()
            outputs = input_function(*args, **kwargs)
            stats = pstats.Stats(self.profiler)
            func_profiles = stats.get_stats_profile().func_profiles

            # Aggregate useful stats
            for key, value in self.config.tracked_fields.items():
                self.agg_stats[key] = func_profiles[value].cumtime / int(
                    func_profiles[value].ncalls
                )

            if self.logfile is None:
                self.profiler.print_stats()
            else:
                # Only need to create stats when dumping, as otherwise this will work.
                self.profiler.create_stats()
                self.profiler.dump_stats(self.logfile)
            self.profiler.disable()
            return outputs

        return perf_wrapper
