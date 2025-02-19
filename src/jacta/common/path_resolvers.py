from pathlib import Path


def get_package_path() -> str:
    """Get the source tree path.

    Returns:
        The source tree path.
    """
    return str(Path(__file__).resolve().parent.parents[2])
