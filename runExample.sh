echo "Compiling example.c"

# Compile the example
make example

# Check if make succeeded
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

echo "Running example..."
./bin/example