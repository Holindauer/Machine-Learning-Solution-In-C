CC=gcc
CFLAGS=-I include
SRC_DIR=src
TEST_DIR=test
BIN_DIR=bin
EXAMPLE_DIR=example
EXAMPLE_TARGET=$(BIN_DIR)/example
EXAMPLE_SRC=$(EXAMPLE_DIR)/example.c
LIB_SOURCES=$(wildcard $(SRC_DIR)/*.c)

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

all: example test_autoGrad test_hashTable test_loadData test_mlp test_forward test_sgd test_valueTracker

# Example target
example: $(EXAMPLE_SRC) $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(EXAMPLE_TARGET)

# Test targets
test_autoGrad: $(TEST_DIR)/test_autoGrad.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_hashTable: $(TEST_DIR)/test_hashTable.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_loadData: $(TEST_DIR)/test_loadData.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_mlp: $(TEST_DIR)/test_mlp.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_forward: $(TEST_DIR)/test_forward.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_sgd: $(TEST_DIR)/test_sgd.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

test_valueTracker: $(TEST_DIR)/test_valueTracker.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)

clean:
	rm -f $(BIN_DIR)/*

