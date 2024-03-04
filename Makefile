CC=gcc
CFLAGS=-I include
LDFLAGS=-lm # Add linker flags here, bc including the math library
SRC_DIR=src
TEST_DIR=test
BIN_DIR=bin
EXAMPLE_DIR=example
LIB_SOURCES=$(wildcard $(SRC_DIR)/*.c)
EXAMPLE_SOURCES=$(wildcard $(EXAMPLE_DIR)/*.c)

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

all: test_autoGrad test_graphStack test_hashTable test_mlp test_forward test_gradientDescent test_loss example_autoGrad example_nn

# Test Targets
test_autoGrad: $(TEST_DIR)/test_autoGrad.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F) $(LDFLAGS)

test_graphStack: $(TEST_DIR)/test_graphStack.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F) $(LDFLAGS)

test_hashTable: $(TEST_DIR)/test_hashTable.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F) $(LDFLAGS)

test_mlp: $(TEST_DIR)/test_mlp.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F) $(LDFLAGS)

test_forward: $(TEST_DIR)/test_forward.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F) $(LDFLAGS)

test_gradientDescent: $(TEST_DIR)/test_gradientDescent.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F) $(LDFLAGS)

test_loss: $(TEST_DIR)/test_loss.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F) $(LDFLAGS)

# Example Targets
example_autoGrad: $(EXAMPLE_DIR)/autoGradExample.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/example_autoGrad $(LDFLAGS)
 
example_nn: $(EXAMPLE_DIR)/nnExample.c $(EXAMPLE_DIR)/loadData.c $(EXAMPLE_DIR)/loadData.h $(EXAMPLE_DIR)/accuracy.c $(EXAMPLE_DIR)/accuracy.h $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/example_nn $(LDFLAGS)