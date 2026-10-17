#!/usr/bin/env python
"""Thin wrapper around egbinn.scripts.shap_analysis for direct invocation.

Equivalent to `python -m egbinn.scripts.shap_analysis` or the `egbinn-shap`
console entry point.
"""

from egbinn.scripts.shap_analysis import main

if __name__ == "__main__":
    main()
