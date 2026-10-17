"""Access to Roboflow's soccer example, third_party/sports/examples/soccer/main.py.

The example is a script, not an installed package, so it is loaded from its
file. Its model paths, class ids, colors, annotators and helper functions are
then reused as they are.
"""

import importlib.util
from functools import lru_cache
from types import ModuleType

from src import config


@lru_cache(maxsize=1)
def load() -> ModuleType:
    """Import the example's main.py as a module, without running it."""
    path = config.SOCCER_EXAMPLE_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("roboflow_soccer_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
