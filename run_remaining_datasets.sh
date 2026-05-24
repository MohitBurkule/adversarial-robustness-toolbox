#!/bin/bash
# run_remaining_datasets.sh
# Main orchestrator to run the remaining 4 datasets: SVHN, STL-10, KMNIST, and EuroSAT.

set -e

# Ensure we are in the correct directory
cd "$(dirname "$0")"

echo "=== Starting Remaining Multi-Dataset Benchmark ==="
echo "Started at: $(date)"

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

echo "=== All Remaining Datasets Complete! ==="
echo "Finished at: $(date)"
