#!/usr/bin/env bash
set -euo pipefail

echo "=== Analyzing Node 1 ==="
./predictor analyze logs/dispatcher_node1.csv --paths 3 --order 3

echo ""
echo "=== Analyzing Node 2 ==="
./predictor analyze logs/dispatcher_node2.csv --paths 3 --order 3

echo ""
echo "=== Cross Analysis ==="
./predictor cross logs/dispatcher_node1.csv logs/dispatcher_node2.csv --paths 3 --order 3 --M 1

