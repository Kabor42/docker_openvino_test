#!/usr/bin/env bash

source /opt/intel/openvino_2022/setupvars.sh
echo
lsusb
echo
python3 sync_run.py -s 2021_10_20_13_45.mp4
