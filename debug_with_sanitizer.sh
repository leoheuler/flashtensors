#!/bin/bash

# NVIDIA Compute Sanitizer Debug Script for flashtensors
# Usage: ./debug_with_sanitizer.sh [memcheck|racecheck|synccheck|initcheck] [script.py]
# chmod +x debug_with_sanitizer.sh

# ./debug_with_sanitizer.sh memcheck ./examples/paligemma_example.py

set -e

CUDA_PATH="/usr/local/cuda-12.8"
COMPUTE_SANITIZER="${CUDA_PATH}/bin/compute-sanitizer"

# Default tool is memcheck
TOOL=${1:-memcheck}
SCRIPT=${2:-examples/paligemma_example.py}

echo "🔍 Running Compute Sanitizer with tool: ${TOOL}"
echo "📄 Script: ${SCRIPT}"

# Build with debug symbols first
echo "🔨 Building with debug symbols..."
DEBUG_BUILD=1 pip install -e . --force-reinstall --no-deps

# Create logs directory
mkdir -p logs

case $TOOL in
    "memcheck")
        echo "🧠 Running Memory Checker..."
        $COMPUTE_SANITIZER \
            --tool memcheck \
            --log-file logs/memcheck_%p.log \
            --check-device-heap yes \
            --force-blocking-launches \
            --generate-coredump \
            python $SCRIPT
        ;;
    "racecheck")
        echo "🏃 Running Race Checker..."
        $COMPUTE_SANITIZER \
            --tool racecheck \
            --log-file logs/racecheck_%p.log \
            --force-blocking-launches \
            python $SCRIPT
        ;;
    "synccheck")
        echo "🔄 Running Sync Checker..."
        $COMPUTE_SANITIZER \
            --tool synccheck \
            --log-file logs/synccheck_%p.log \
            --force-blocking-launches \
            python $SCRIPT
        ;;
    "initcheck")
        echo "🎯 Running Init Checker..."
        $COMPUTE_SANITIZER \
            --tool initcheck \
            --log-file logs/initcheck_%p.log \
            --force-blocking-launches \
            python $SCRIPT
        ;;
    *)
        echo "❌ Unknown tool: $TOOL"
        echo "Available tools: memcheck, racecheck, synccheck, initcheck"
        exit 1
        ;;
esac

echo "✅ Compute Sanitizer analysis complete!"
echo "📊 Check logs in: logs/${TOOL}_*.log"

# Show summary of any issues found
echo "📋 Summary of issues found:"
if ls logs/${TOOL}_*.log 1> /dev/null 2>&1; then
    grep -h "ERROR\|WARN\|========" logs/${TOOL}_*.log | head -20 || echo "No errors/warnings found in logs"
else
    echo "No log files generated"
fi
