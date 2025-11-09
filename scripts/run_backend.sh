#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

./backend --bind 127.0.0.1:6001 --run 0 --trial 0 --log logs/backend_1.csv &
BACKEND1_PID=$!
./backend --bind 127.0.0.1:6002 --run 0 --trial 0 --log logs/backend_2.csv &
BACKEND2_PID=$!
./backend --bind 127.0.0.1:6003 --run 0 --trial 0 --log logs/backend_3.csv &
BACKEND3_PID=$!

trap "kill $BACKEND1_PID $BACKEND2_PID $BACKEND3_PID 2>/dev/null || true" EXIT

wait

