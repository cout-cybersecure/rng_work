#!/usr/bin/env bash
set -euo pipefail

for nic in /sys/class/net/*; do
    ifname=$(basename "$nic")
    if [ "$ifname" = "lo" ]; then
        continue
    fi
    if [ -d "$nic" ]; then
        echo "Disabling offloads on $ifname"
        ethtool -K "$ifname" gro off 2>/dev/null || true
        ethtool -K "$ifname" gso off 2>/dev/null || true
        ethtool -K "$ifname" tso off 2>/dev/null || true
        ethtool -K "$ifname" lro off 2>/dev/null || true
    fi
done

for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$cpu" ]; then
        echo "performance" > "$cpu"
    fi
done

