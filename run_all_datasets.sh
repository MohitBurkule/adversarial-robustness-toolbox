#!/bin/bash
# run_all_datasets.sh
# Main orchestrator to run all 50 hypotheses on CIFAR-10 and then on Imagenette.

set -e

# Ensure we are in the correct directory
cd "$(dirname "$0")"

echo "=== Starting Complete Multi-Dataset Benchmark ==="
echo "Started at: $(date)"

# 1. Run on CIFAR-10
echo "=== Step 1/2: Running on CIFAR-10 ==="
./run_patch_benchmark.sh cifar10

# 2. Run on Imagenette
echo "=== Step 2/2: Running on Imagenette ==="
./run_patch_benchmark.sh imagenette

echo "=== All Datasets Complete! ==="
echo "Finished at: $(date)"
