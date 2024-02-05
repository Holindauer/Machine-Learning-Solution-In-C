#!/bin/bash

# Compile the tests
echo "Compiling tests..."
make

# Check if make succeeded
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

echo "Running tests..."

# Define your test binaries here
tests=("test_autoGrad" "test_hashTable" "test_loadData" "test_mlp")

# Directory where binaries are located
BIN_DIR="bin"

# Iterate over the tests array and execute each test
for test in "${tests[@]}"; do
  echo "Running $test..."
  ./$BIN_DIR/$test
  if [ $? -ne 0 ]; then
    echo "$test failed."
    exit 1
  else
    echo "$test passed."
  fi
done

echo "All tests passed successfully."
