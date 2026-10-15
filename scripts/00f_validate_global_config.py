#!/usr/bin/env python3
"""
00f_validate_global_config.py
Validate the global config registry and print a human-readable summary.

Usage (on the compute cluster):
    env/run_py.sh scripts/00f_validate_global_config.py
"""

from hfpef_dtr.config import main

if __name__ == "__main__":
    main()
