#!/bin/bash

# Simple Compile Script

# Check if an input file was provided
if [ -z "$1" ]; then
  echo "Usage: ./compile.sh <input_file.mp>"
  exit 1
fi

INPUT_FILE="$1"
BASENAME=$(basename "$INPUT_FILE" .mp)
OUTPUT_C_FILE="${BASENAME}.c"
# Use the basename of the input for the final executable, not the full path
OUTPUT_EXEC="${BASENAME}"
TEMP_C_FILE="_temp_output.c" # Temporary file for C code
TEMP_OBJ_FILE="_temp_output.o" # Temporary file for object code

# Ensure the temporary files are cleaned up on exit (success or error)
trap 'rm -f "$TEMP_C_FILE" "$TEMP_OBJ_FILE"' EXIT

echo "--- Compiling $INPUT_FILE to C ---"
# Run the Python compiler, redirecting its output (C code) to the temp file
# Use python -m to run as module, allowing relative imports
if python -m src.compiler "$INPUT_FILE" --output "$TEMP_C_FILE" --compiler-log-level DEBUG; then
  echo "Python compilation successful. C code written to $TEMP_C_FILE"
else
  echo "Error: Python compiler (src.compiler) failed."
  exit 1
fi

# Optional: Display generated C code (for debugging)
# echo "--- Generated C Code ($TEMP_C_FILE) ---"
# cat "$TEMP_C_FILE"
# echo "------------------------------------"

echo "--- Compiling C code ($TEMP_C_FILE) to executable ($OUTPUT_EXEC) ---"
# Compile the C code to an object file first
if gcc -Wall -Wextra -std=c11 -c "$TEMP_C_FILE" -o "$TEMP_OBJ_FILE"; then
    echo "C compilation to object file successful ($TEMP_OBJ_FILE)"
    # Link the object file to create the final executable
    if gcc "$TEMP_OBJ_FILE" -o "$OUTPUT_EXEC"; then
        echo "C linking successful. Executable created: $OUTPUT_EXEC"
    else
        echo "Error: C linker (gcc) failed."
        exit 1
    fi
else
  echo "Error: C compiler (gcc) failed during compilation to object file."
  exit 1
fi

echo "--- Compilation complete for $INPUT_FILE -> $OUTPUT_EXEC ---"
exit 0 