#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./monitor.sh <path_to_config.cfg>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_FILE="$(readlink -f "$1")"
export OUT_DIR="$SCRIPT_DIR/output"

run_aggregation() {
    source "$CONFIG_FILE"
    clear
    echo "======================================================================="
    echo "            HAMILTON DECENTRALIZED PERFORMANCE MONITOR                 "
    echo "======================================================================="
    
    if [ -f "$OUT_DIR/result.profile.txt" ]; then
        echo "Profile Generation Run: $(cat $OUT_DIR/result.profile.txt) seconds"
    else
        echo "Profile Generation Run: [ RUNNING / QUEUED ]"
    fi
    echo "-----------------------------------------------------------------------"
    printf "%-10s | %-15s | %-15s\n" "Trial" "Normal Time" "Boost Time"
    echo "-----------------------------------------------------------------------"

    norm_total=0; norm_count=0
    boost_total=0; boost_count=0

    for i in $(seq 1 $NUM_TRIALS); do
        norm_file="$OUT_DIR/result.normal.${i}.txt"
        boost_file="$OUT_DIR/result.boost.${i}.txt"
        
        n_val="--"
        b_val="--"
        
        if [ -f "$norm_file" ]; then
            n_val="$(cat $norm_file)s"
            norm_total=$(echo "$norm_total + $(cat $norm_file)" | bc)
            norm_count=$((norm_count + 1))
        fi
        
        if [ -f "$boost_file" ]; then
            b_val="$(cat $boost_file)s"
            boost_total=$(echo "$boost_total + $(cat $boost_file)" | bc)
            boost_count=$((boost_count + 1))
        fi
        
        printf "%-10s | %-15s | %-15s\n" "Trial $i" "$n_val" "$b_val"
    done

    echo "-----------------------------------------------------------------------"
    
    if [ $norm_count -gt 0 ]; then
        norm_avg=$(echo "scale=2; $norm_total / $norm_count" | bc)
        echo "Normal Running Avg ($norm_count/$NUM_TRIALS completed): ${norm_avg}s"
    else
        echo "Normal Running Avg: Waiting for first completion..."
    fi

    if [ $boost_count -gt 0 ]; then
        boost_avg=$(echo "scale=2; $boost_total / $boost_count" | bc)
        echo "Boost Running Avg  ($boost_count/$NUM_TRIALS completed): ${boost_avg}s"
    fi

    if [ $norm_count -gt 0 ] && [ $boost_count -gt 0 ]; then
        diff_pct=$(echo "scale=2; (($norm_avg - $boost_avg) / $norm_avg) * 100" | bc)
        echo "Current Boost Performance Improvement: ${diff_pct}%"
    fi
    echo "======================================================================="
}

export -f run_aggregation

watch -n 10 bash -c run_aggregation
