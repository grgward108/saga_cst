#!/bin/bash

# Check if exp_name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <exp_name>"
    exit 1
fi

exp_name=$1
objects=("toothpaste" "camera" "wineglass" "fryingpan" "binoculars" "mug")

for object in "${objects[@]}"; do
    python opt_grasppose.py --exp_name $exp_name --gender male --pose_ckpt_path logs/GraspPose/change_decoder_10202024/snapshots/TR00_E050_net.pt --object $object --n_object_samples 25
    if [ $? -ne 0 ]; then
        echo "Command failed for object: $object"
        exit 1
    fi
done