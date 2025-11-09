#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Same Seed Two Nodes Experiment ==="
bash scripts/run_same_seed_two_nodes.sh

echo ""
echo "=== Running Quantum RNG Two Nodes Experiment ==="
bash scripts/run_qrng_two_nodes.sh

echo ""
echo "=== All experiments completed ==="

