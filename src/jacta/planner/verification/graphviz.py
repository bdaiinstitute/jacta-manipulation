# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from graphviz import Source
from pydrake.systems.framework import System


def render_system_with_graphviz(
    system: System, output_file: str = "system_view.gz"
) -> None:
    """
    Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file.
    """

    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)
