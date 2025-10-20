#!/bin/bash

# GDB Debug Script for flashtensors
# Usage: ./debug_with_gdb.sh [script.py]
# chmod +x debug_with_gdb.sh

set -e

SCRIPT=${1:-examples/paligemma_example.py}

echo "🔍 GDB Debugging Script for flashtensors"
echo "📄 Script: ${SCRIPT}"

# Build with debug symbols first
echo "🔨 Building with debug symbols..."
DEBUG_BUILD=1 pip install -e . --force-reinstall --no-deps

# Enable core dumps
echo "💾 Enabling core dumps..."
ulimit -c unlimited

# Create logs directory
mkdir -p logs

echo ""
echo "🚀 Starting GDB analysis..."
echo ""

# Create GDB command file for automated debugging
cat > logs/gdb_commands.txt << 'EOF'
# Set up GDB for Python debugging
set pagination off
set logging on logs/gdb_output.log
set logging overwrite on

# Handle signals
handle SIGSEGV stop print
handle SIGABRT stop print

# Run the program
run

# When it crashes, get detailed info
echo === CRASH DETECTED ===
bt
echo === REGISTERS ===
info registers
echo === LOCALS ===
info locals
echo === ARGUMENTS ===
info args
echo === MEMORY MAP ===
info proc mappings
echo === THREADS ===
info threads
echo === CURRENT FRAME ===
frame

# Try to get more context
echo === STACK FRAMES ===
bt 20

# If we can, show source
echo === SOURCE CONTEXT ===
list

# Continue to see if there are more crashes
continue

# Quit when done
quit
EOF

echo "Running GDB with automated crash analysis..."
echo "This will:"
echo "  1. Run your program under GDB"
echo "  2. Automatically catch any segfaults"
echo "  3. Collect detailed crash information"
echo "  4. Save everything to logs/gdb_output.log"
echo ""

# Run GDB with the command file
gdb --batch --command=logs/gdb_commands.txt --args python ${SCRIPT}

echo ""
echo "✅ GDB analysis complete!"
echo ""
echo "📊 Results saved to:"
echo "  - logs/gdb_output.log (detailed GDB output)"
echo ""

# Show crash summary
if [ -f logs/gdb_output.log ]; then
    echo "📋 Crash Summary:"
    echo "=================="

    # Extract key crash information
    echo ""
    echo "🎯 Crash Location:"
    grep -A 5 "Program received signal" logs/gdb_output.log || echo "No crash detected"

    echo ""
    echo "📚 Stack Trace:"
    sed -n '/=== CRASH DETECTED ===/,/=== REGISTERS ===/p' logs/gdb_output.log | grep "^#" | head -10 || echo "No stack trace available"

    echo ""
    echo "🔍 For full details, check: logs/gdb_output.log"
else
    echo "❌ No GDB output file generated"
fi

# Check for core dumps
echo ""
echo "💾 Checking for core dumps..."
if ls core.* 1> /dev/null 2>&1; then
    echo "📦 Core dump(s) found:"
    ls -la core.*
    echo ""
    echo "💡 To analyze manually:"
    echo "   gdb python core.*"
else
    echo "📝 No core dumps found (this is normal with GDB)"
fi

echo ""
echo "🎉 Analysis complete! Check logs/gdb_output.log for details."