#!/bin/bash

# Stop if any error occurs
set -e


# User has to specify GPU ID
if [ -z "$1" ]; then
	echo "ERROR: You have to specify one GPU ID from (0, 1, 2, 4)"
	echo "Please try: ./run_roberta_training.sh <0, 1, 2, 4>"
	exit 1
fi

# Mapping the short GPU number to specific GPU ID
case $1 in
	0)
		REAL_GPU_ID="GPU-cf012594-4399-a8c2-1d9e-4f182cd9fcbe"
		;;
	1)
		REAL_GPU_ID="GPU-f17911a9-8dca-010e-b80a-417f8bb71a51"
		;;
	2)
		REAL_GPU_ID="GPU-4766ef07-3fcc-8bb3-c22c-d3c97d363c23"
		;;
	4)
		REAL_GPU_ID="GPU-96f047f6-0272-02db-e089-678debe69d9b"
		;;
	*)	# If any other number has been given
		echo "ERROR: Unvalid GPU ID given."
		echo "-- GPU ID has to be in (0, 1, 2, 4)"
		exit 1
		;;
esac

echo "Checking status of GPU $1 (UUID: $REAL_GPU_ID)..."

PIDS=$(nvidia-smi -i "$REAL_GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

if [ -n "$PIDS" ]; then
	echo "ABORTED! Some processes are running on specified GPU $1: (PIDs: $PIDS)."
	echo "-- Script is aborted since processes indicate that specified GPU is in use!"
    	exit 1
fi

echo "GPU $1 seems to be free! Starting script on specified GPU=$1..."

export CUDA_VISIBLE_DEVICES="$REAL_GPU_ID"

nice -n 10 uv run python -m scripts.run_evaluation --eval-roberta --set-checkpoint=experiments/roberta_run_2/checkpoint-109000 | tee roberta_eval_log.txt
