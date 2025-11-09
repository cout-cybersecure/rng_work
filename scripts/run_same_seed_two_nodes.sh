#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

./backend --bind 127.0.0.1:6001 --run 0 --trial 0 --log logs/backend_node1_1.csv &
BACKEND1_1_PID=$!
./backend --bind 127.0.0.1:6002 --run 0 --trial 0 --log logs/backend_node1_2.csv &
BACKEND1_2_PID=$!
./backend --bind 127.0.0.1:6003 --run 0 --trial 0 --log logs/backend_node1_3.csv &
BACKEND1_3_PID=$!

./backend --bind 127.0.0.1:7001 --run 0 --trial 0 --log logs/backend_node2_1.csv &
BACKEND2_1_PID=$!
./backend --bind 127.0.0.1:7002 --run 0 --trial 0 --log logs/backend_node2_2.csv &
BACKEND2_2_PID=$!
./backend --bind 127.0.0.1:7003 --run 0 --trial 0 --log logs/backend_node2_3.csv &
BACKEND2_3_PID=$!

sleep 1

./dispatcher --bind 127.0.0.1:5000 \
    --backends 127.0.0.1:6001,127.0.0.1:6002,127.0.0.1:6003 \
    --jitter_ns 100000 \
    --rng drbg \
    --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
    --nonce 000000000000000000000000 \
    --run 0 \
    --trial 0 \
    --log logs/dispatcher_node1.csv &
DISPATCHER1_PID=$!

./dispatcher --bind 127.0.0.1:5001 \
    --backends 127.0.0.1:7001,127.0.0.1:7002,127.0.0.1:7003 \
    --jitter_ns 100000 \
    --rng drbg \
    --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
    --nonce 000000000000000000000000 \
    --run 0 \
    --trial 0 \
    --log logs/dispatcher_node2.csv &
DISPATCHER2_PID=$!

sleep 1

./generator --dst 127.0.0.1:5000 \
    --requests 100000 \
    --interval_us 100 \
    --run 0 \
    --trial 0 \
    --log logs/generator_node1.csv &
GENERATOR1_PID=$!

./generator --dst 127.0.0.1:5001 \
    --requests 100000 \
    --interval_us 100 \
    --run 0 \
    --trial 0 \
    --log logs/generator_node2.csv &
GENERATOR2_PID=$!

wait $GENERATOR1_PID $GENERATOR2_PID

kill $DISPATCHER1_PID $DISPATCHER2_PID 2>/dev/null || true
wait $DISPATCHER1_PID $DISPATCHER2_PID 2>/dev/null || true

kill $BACKEND1_1_PID $BACKEND1_2_PID $BACKEND1_3_PID \
     $BACKEND2_1_PID $BACKEND2_2_PID $BACKEND2_3_PID 2>/dev/null || true

wait

