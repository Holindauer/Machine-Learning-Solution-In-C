echo "Compiling nnExample.c"

# Compile the example
make example_nn

# Check if make succeeded
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

echo "Running neural net training example..."
./bin/example_nn