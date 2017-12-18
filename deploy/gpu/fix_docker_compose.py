#!/usr/bin/env python
"""
Add nvidia-docker as a default run time
https://github.com/NVIDIA/nvidia-docker/issues/568
"""
import json
with open('/etc/docker/daemon.json') as f:
    j = json.load(f)
j["default-runtime"] = "nvidia"
with open('/etc/docker/daemon.json','w') as f:
    json.dump(j,f)