#!/bin/bash
# run_all_datasets.sh
# Main orchestrator to run all 50 hypotheses on CIFAR-10 and then on Imagenette.

set -e

# Ensure we are in the correct directory
cd "$(dirname "$0")"

echo "=== Starting Complete Multi-Dataset Benchmark ==="
echo "Started at: $(date)"

# 1. Run on CIFAR-10
echo "=== Step 1/6: Running on CIFAR-10 ==="
./run_patch_benchmark.sh cifar10

# 2. Run on Imagenette
echo "=== Step 2/6: Running on Imagenette ==="
./run_patch_benchmark.sh imagenette

# 3. Run on SVHN
echo "=== Step 3/6: Running on SVHN ==="
./run_patch_benchmark.sh svhn

# 4. Run on STL-10
echo "=== Step 4/6: Running on STL-10 ==="
./run_patch_benchmark.sh stl10

# 5. Run on KMNIST
echo "=== Step 5/6: Running on KMNIST ==="
./run_patch_benchmark.sh kmnist

# 6. Run on EuroSAT
echo "=== Step 6/6: Running on EuroSAT ==="
./run_patch_benchmark.sh eurosat

echo "=== All 6 Datasets Complete! ==="
echo "Finished at: $(date)"
