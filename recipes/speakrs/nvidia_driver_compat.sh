#!/usr/bin/env bash

# keep new CUDA processes aligned with the host kernel driver after package updates
nvidia_driver_compat_dir="${NVIDIA_DRIVER_COMPAT_DIR:-/root/code/nvidia-580.95.05/root/usr/lib/x86_64-linux-gnu}"
if [[ -d "$nvidia_driver_compat_dir" ]]; then
    export LD_LIBRARY_PATH="$nvidia_driver_compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
