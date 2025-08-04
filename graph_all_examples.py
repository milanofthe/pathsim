import sys
import importlib.util
from pathlib import Path

from pathsim.graphing import graph

examples_dir = Path(__file__).parent / "examples"

for example_file in examples_dir.glob("*.py"):
    # Skip __init__.py or any non-example files if needed
    if example_file.name.startswith("__"):
        continue

    # Dynamically import the example module
    spec = importlib.util.spec_from_file_location(example_file.stem, example_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[example_file.stem] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Failed to import {example_file.name}: {e}")
        continue

    # Check for blocks and connections
    if hasattr(module, "blocks") and hasattr(module, "connections"):
        print(f"Graphing {example_file.name} ...")
        try:
            graph(module.blocks, module.connections, file_name=example_file.stem)
        except Exception as e:
            print(f"Failed to graph {example_file.name}: {e}")
    else:
        print(f"{example_file.name} does not define 'blocks' and 'connections'. Skipping.")