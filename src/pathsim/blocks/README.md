# Block Library

This is the place where the blocks are defined. All blocks inherent bas `Block` class from `_block.py` and implement the specific methods for the block behavior. The blocks are grouped thematically into modules. Standard blocks can be imported directly from the block library like this:

```python
from pathsim.blocks import Adder, Amplifier
```

Or from the respective module:

```python
from pathsim.blocks.adder import Adder
from pathsim.blockd.amplifier import Amplifier
```

---

The goal is to keep the available blocks that can be imported direcly general purpose and have separate toolboxes for special purpose blocks. For example for different domains such as chemical engineering, or robotics.