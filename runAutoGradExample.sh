echo "Compiling autoGradExample.c"

# Compile the example
make example_autoGrad

# Check if make succeeded
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

echo "Running autograd example..."
./bin/example_autoGrad