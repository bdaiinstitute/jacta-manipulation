# mujoco_extensions/__init__.py
import importlib.metadata  # Recommended for finding package version etc.
import os
import platform
import warnings
from pathlib import Path

import mujoco

# Optional: Get package version
try:
    __version__ = importlib.metadata.version("mujoco-extensions")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"  # Fallback if not installed properly

print(f"Loading mujoco-extensions version {__version__}")

# --- MuJoCo Plugin Discovery ---


def _find_and_prepare_mujoco_plugin_path():
    """Finds the bundled MuJoCo plugin(s) and updates MUJOCO_PLUGIN_PATH."""
    plugin_dir_str = "unknown"  # Initialize for robustness in case of early errors
    try:
        # Use Path(__file__).parent which points to the directory of this __init__.py
        package_dir = Path(__file__).parent.resolve()

        # This is the subdirectory name we used in the CMake install DESTINATION
        plugin_subdir_name = "mujoco_plugins"
        plugin_dir = package_dir / plugin_subdir_name

        # --- FIX: Assign plugin_dir_str *before* the check ---
        plugin_dir_str = str(plugin_dir)

        # Now check if the directory exists
        if plugin_dir.is_dir():
            print(
                f"[mujoco-extensions] Found MuJoCo plugin directory: {plugin_dir_str}"
            )

            # Get the current MUJOCO_PLUGIN_PATH, or an empty string if not set
            current_path = os.environ.get("MUJOCO_PLUGIN_PATH", "")
            # Split path, handling potential empty string case
            path_list = [p for p in current_path.split(os.pathsep) if p]

            # Add our plugin directory if it's not already there
            if plugin_dir_str not in path_list:
                # Prepend our path
                new_path_list = [plugin_dir_str] + path_list
                new_path = os.pathsep.join(new_path_list)
                os.environ["MUJOCO_PLUGIN_PATH"] = new_path
                print(
                    f"[mujoco-extensions] Updated MUJOCO_PLUGIN_PATH: {os.environ['MUJOCO_PLUGIN_PATH']}"
                )
            else:
                print(
                    f"[mujoco-extensions] Plugin path '{plugin_dir_str}' already in MUJOCO_PLUGIN_PATH."
                )

            # --- Optional: Check if the actual library file exists ---
            # Use the CMake target name to construct expected library names
            # !!! DOUBLE CHECK this matches your add_library() name in CMakeLists.txt !!!
            cmake_target_name = "dynamic_pd_actuator_plugin"  # Check if this is exactly 'dynamic_pd_actuator_plugin' in CMake
            prefix = ""
            suffix = ""
            if platform.system() == "Linux":
                prefix = "lib"
                suffix = ".so"
            elif platform.system() == "Darwin":  # macOS
                prefix = "lib"
                suffix = ".dylib"
            elif platform.system() == "Windows":
                prefix = ""  # Usually no 'lib' prefix on Windows for DLLs
                suffix = ".dll"

            if suffix:  # Only check if we know the expected suffix
                expected_lib_file = plugin_dir / f"{prefix}{cmake_target_name}{suffix}"
                if not expected_lib_file.is_file():
                    warnings.warn(
                        f"[mujoco-extensions] MuJoCo plugin directory exists, but expected library "
                        f"'{expected_lib_file.name}' was not found inside {plugin_dir_str}. "
                        "Plugin loading may fail. Check CMake install rules (DESTINATION should be "
                        f"'{plugin_subdir_name}'), target name ('{cmake_target_name}'), and packaging.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    print(
                        f"[mujoco-extensions] Verified plugin library '{expected_lib_file.name}' exists."
                    )
            # --- End Optional Check ---
            print("Loading plugin libraries.")
            mujoco.mj_loadAllPluginLibraries(plugin_dir_str)

        else:
            # Directory does not exist, but plugin_dir_str is now defined
            print(
                # Now safe to use plugin_dir_str here
                f"[mujoco-extensions] Optional MuJoCo plugin directory '{plugin_dir_str}' not found. "
                "No bundled MuJoCo plugins detected. Ensure CMake installed the plugin to the correct relative path."
            )
            # warnings.warn(...) # This would also be safe now if uncommented

    except Exception as e:
        # This warning message is safe as it doesn't rely on potentially unassigned variables from the try block
        warnings.warn(
            f"[mujoco-extensions] Error during MuJoCo plugin path setup: {e}",
            RuntimeWarning,
            stacklevel=2,
        )


# --- IMPORTANT: Execute the setup when the package is imported ---
_find_and_prepare_mujoco_plugin_path()
# ---

# Your package's regular imports and code can follow...
# Example: make submodules easily importable
# from .jacobian_smoothing import _jacobian_smoothing
# from .policy_rollout import _policy_rollout

print("[mujoco-extensions] Package initialized.")
