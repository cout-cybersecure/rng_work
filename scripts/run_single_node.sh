#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

./backend --bind 127.0.0.1:6001 --run 0 --trial 0 --log logs/backend_1.csv >/dev/null 2>&1 &
BACKEND1_PID=$!
./backend --bind 127.0.0.1:6002 --run 0 --trial 0 --log logs/backend_2.csv >/dev/null 2>&1 &
BACKEND2_PID=$!
./backend --bind 127.0.0.1:6003 --run 0 --trial 0 --log logs/backend_3.csv >/dev/null 2>&1 &
BACKEND3_PID=$!

sleep 1

./dispatcher --bind 127.0.0.1:5000 \
    --backends 127.0.0.1:6001,127.0.0.1:6002,127.0.0.1:6003 \
    --jitter_ns 100000 \
    --rng drbg \
    --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
    --nonce 000000000000000000000000 \
    --run 0 \
    --trial 0 \
    --log logs/dispatcher.csv >/dev/null 2>&1 &
DISPATCHER_PID=$!

sleep 1

./generator --dst 127.0.0.1:5000 \
    --requests 100000 \
    --interval_us 100 \
    --run 0 \
    --trial 0 \
    --log logs/generator.csv

kill $DISPATCHER_PID 2>/dev/null || true
wait $DISPATCHER_PID 2>/dev/null || true

kill $BACKEND1_PID $BACKEND2_PID $BACKEND3_PID 2>/dev/null || true
wait $BACKEND1_PID $BACKEND2_PID $BACKEND3_PID 2>/dev/null || true

