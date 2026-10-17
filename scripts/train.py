#!/usr/bin/env python
"""Thin wrapper around egbinn.scripts.train for direct invocation.

Equivalent to `python -m egbinn.scripts.train` or the `egbinn-train`
console entry point.
"""

from egbinn.scripts.train import main

if __name__ == "__main__":
    main()
