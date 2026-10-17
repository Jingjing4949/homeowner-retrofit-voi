"""H-A6 on dataset B (ResStock 2024.2); see h_a6.py."""
import sys

import h_a6

if __name__ == "__main__":
    assert sys.argv[1] in ("smoke", "run")
    h_a6.run("B", sys.argv[1] == "smoke")
