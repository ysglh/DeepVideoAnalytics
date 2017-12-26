#!/usr/bin/env python
import sys, shutil, os

if __name__ == "__main__":
    if sys.platform == 'darwin':
        shutil.copy("../configs/custom_defaults/defaults_mac.py", 'dvaui/defaults.py')
    elif os.path.isfile("../configs/custom_defaults/defaults.py"):
        shutil.copy("../configs/custom_defaults/defaults.py", 'dvaui/defaults.py')
    else:
        raise ValueError("defaults.py not found, if you have mounted a custom_config volume")
