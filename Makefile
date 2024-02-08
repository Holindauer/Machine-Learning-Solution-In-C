CC=gcc
CFLAGS=-I include
SRC_DIR=src
TEST_DIR=test
BIN_DIR=bin

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

all: main test_autoGrad test_hashTable test_loadData  test_mlp test_forward test_sgd

main: $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/main $(SRC_DIR)/main.c

# Test targets (separate executables for each test)
test_autoGrad: $(TEST_DIR)/test_autoGrad.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_autoGrad $(TEST_DIR)/test_autoGrad.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c

test_hashTable: $(TEST_DIR)/test_hashTable.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/autoGrad.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_hashTable $(TEST_DIR)/test_hashTable.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/autoGrad.c

test_loadData: $(TEST_DIR)/test_loadData.c $(SRC_DIR)/loadData.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_loadData $(TEST_DIR)/test_loadData.c $(SRC_DIR)/loadData.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c

test_mlp: $(TEST_DIR)/test_mlp.c $(SRC_DIR)/mlp.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/loadData.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_mlp $(TEST_DIR)/test_mlp.c $(SRC_DIR)/mlp.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/loadData.c

test_forward: $(TEST_DIR)/test_forward.c $(SRC_DIR)/mlp.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/loadData.c $(SRC_DIR)/forward.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_forward $(TEST_DIR)/test_forward.c $(SRC_DIR)/mlp.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/loadData.c $(SRC_DIR)/forward.c

test_sgd: $(TEST_DIR)/test_sgd.c $(SRC_DIR)/mlp.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/loadData.c $(SRC_DIR)/forward.c $(SRC_DIR)/sgd.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_sgd $(TEST_DIR)/test_sgd.c $(SRC_DIR)/mlp.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/loadData.c $(SRC_DIR)/forward.c $(SRC_DIR)/sgd.c

clean:
	rm -f $(BIN_DIR)/*
