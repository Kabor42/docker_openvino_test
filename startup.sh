#!/usr/bin/env bash

source /opt/intel/openvino_2022/setupvars.sh
echo
lsusb
echo
python3 /app/sync_run.py
