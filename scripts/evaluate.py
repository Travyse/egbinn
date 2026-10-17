#!/usr/bin/env python
"""Thin wrapper around egbinn.scripts.evaluate for direct invocation.

Equivalent to `python -m egbinn.scripts.evaluate` or the `egbinn-evaluate`
console entry point.
"""

from egbinn.scripts.evaluate import main

if __name__ == "__main__":
    main()
