CC=gcc
CFLAGS=-I include
SRC_DIR=src
TEST_DIR=test
BIN_DIR=bin
LIB_SOURCES=$(wildcard $(SRC_DIR)/*.c)

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

all: test_autoGrad test_graphStack test_hashTable test_mlp test_forward

# Test targets
test_autoGrad: $(TEST_DIR)/test_autoGrad.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_graphStack: $(TEST_DIR)/test_graphStack.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_hashTable: $(TEST_DIR)/test_hashTable.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_mlp: $(TEST_DIR)/test_mlp.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_forward: $(TEST_DIR)/test_forward.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)