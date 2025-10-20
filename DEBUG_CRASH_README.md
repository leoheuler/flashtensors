# flashtensors Crash Debugging Guide

## Quick Start

### Automated GDB Analysis
```bash
# Make executable and run
chmod +x debug_with_gdb.sh
./debug_with_gdb.sh

# Or specify a different script
./debug_with_gdb.sh examples/paligemma_example.py
```

This will automatically:
- Build with debug symbols
- Run under GDB
- Catch segfaults
- Generate detailed crash report in `logs/gdb_output.log`

---

## Manual Debugging Methods

### Method 1: Interactive GDB Session
```bash
# Build with debug symbols
DEBUG_BUILD=1 pip install -e . --force-reinstall --no-deps

# Enable core dumps
ulimit -c unlimited

# Run under GDB
gdb python
```

**In GDB:**
```gdb
(gdb) run examples/paligemma_example.py
# Wait for crash...

# When it crashes:
(gdb) bt                    # Stack trace
(gdb) info locals          # Local variables
(gdb) info registers       # CPU registers
(gdb) list                 # Source code around crash
(gdb) frame 0              # Examine crash frame
(gdb) print variable_name  # Check specific variables
(gdb) quit                 # Exit
```

### Method 2: Core Dump Analysis
```bash
# Find core dump after crash
ls -la core.*

# Analyze with GDB
gdb python core.12345

# In GDB:
(gdb) bt                   # Stack trace from crash
(gdb) info threads         # All threads at crash time
(gdb) thread 2             # Switch to thread 2
(gdb) bt                   # Stack trace for that thread
```

### Method 3: Compute Sanitizer (Memory Errors)
```bash
./debug_with_sanitizer.sh memcheck
```

---

## Understanding the Output

### Stack Trace Example
```
#0  0x00007fff8b9e1234 in cudaMemcpy () from /usr/local/cuda/lib64/libcudart.so
#1  0x00007fff8b9e5678 in Model::ToGpu() at model.cpp:344
#2  0x00007fff8b9e9abc in DispatchToGpu() at model.cpp:580
```

**This tells you:**
- **Frame #0**: Crash happened in `cudaMemcpy`
- **Frame #1**: Called from `Model::ToGpu()` at line 344 in model.cpp
- **Frame #2**: Which was called from `DispatchToGpu()` at line 580

### Common Crash Locations

#### 1. CUDA Memory Operations
```
#0  cudaMemcpy ()
#1  Model::ToGpu() at model.cpp:344
```
**Likely cause**: Invalid GPU memory pointer, wrong size, or uninitialized CUDA context

#### 2. IPC Handle Issues
```
#0  cudaIpcOpenMemHandle ()
#1  allocate_cuda_memory()
```
**Likely cause**: Corrupted CUDA IPC handle, invalid handle, or permission issues

#### 3. Vector/Container Access
```
#0  std::vector::operator[] ()
#1  DispatchToGpu() at model.cpp:598
```
**Likely cause**: Index out of bounds, uninitialized vector

#### 4. Null Pointer Dereference
```
#0  Model::DispatchToGpu() at model.cpp:575
```
**Likely cause**: `host_ptr_vector_` is null, uninitialized shared_ptr

---

## Common Fixes

### CUDA Memory Issues
```cpp
// Add bounds checking before CUDA calls
if (handle_idx >= device_ptr_list.size()) {
    LOG(ERROR) << "Invalid handle_idx";
    return -1;
}

// Validate pointers before use
if (device_ptr_list[handle_idx] == nullptr) {
    LOG(ERROR) << "Null device pointer";
    return -1;
}
```

### Race Condition Fixes
```cpp
// Wait for initialization
while (!host_ptr_vector_ && retries-- > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
```

### Memory Debugging
```cpp
// Add memory validation
CUDA_CHECK(cudaPointerGetAttributes(&attrs, ptr));
if (attrs.memoryType != cudaMemoryTypeDevice) {
    LOG(ERROR) << "Invalid device memory";
    return -1;
}
```

---

## Useful GDB Commands

### Navigation
```gdb
bt              # Stack trace
frame N         # Go to frame N
up              # Go up one frame
down            # Go down one frame
list            # Show source code
```

### Information
```gdb
info locals     # Local variables
info args       # Function arguments
info registers  # CPU registers
info threads    # All threads
info proc       # Process info
```

### Variables
```gdb
print var_name          # Print variable
print *pointer          # Dereference pointer
print array[0]@10       # Print array elements
set var = value         # Change variable
```

### Memory
```gdb
x/10x address          # Examine 10 hex words at address
x/s string_ptr         # Show string
info proc mappings     # Memory maps
```

---

## Troubleshooting

### No Debug Symbols
If you see `<optimized out>` or no function names:
```bash
# Rebuild with debug symbols
DEBUG_BUILD=1 pip install -e . --force-reinstall --no-deps
```

### Core Dumps Not Generated
```bash
# Check limits
ulimit -c
# Should show "unlimited"

# Set if needed
ulimit -c unlimited
```

### GDB Can't Find Source
```gdb
# In GDB, set source directory
(gdb) directory /path/to/source
```

### Permission Issues
```bash
# Run with proper permissions
sudo sysctl kernel.core_pattern=/tmp/core.%e.%p.%t  # Linux
```

---

## Files Generated

- `logs/gdb_output.log` - Complete GDB session output
- `logs/gdb_commands.txt` - GDB commands used
- `core.*` - Core dump files (if generated)
- `logs/memcheck_*.log` - Compute Sanitizer output

---

## Next Steps After Finding Crash

1. **Identify the exact line** causing the segfault
2. **Check nearby variables** for null pointers or invalid values
3. **Trace the call stack** to understand how you got there
4. **Add validation** before the crashing operation
5. **Test the fix** with the same debugging tools

Remember: The goal is to find the **exact memory access** that's invalid and add proper validation before it happens.