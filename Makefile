CC=gcc
CFLAGS=-I include
SRC_DIR=src
TEST_DIR=test
BIN_DIR=bin

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

all: main test_autoGrad test_hashTable #test_loadData 

main: $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/main $(SRC_DIR)/main.c

# Test targets (separate executables for each test)
test_autoGrad: $(TEST_DIR)/test_autoGrad.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_autoGrad $(TEST_DIR)/test_autoGrad.c $(SRC_DIR)/autoGrad.c $(SRC_DIR)/hashTable.c

test_hashTable: $(TEST_DIR)/test_hashTable.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/autoGrad.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_hashTable $(TEST_DIR)/test_hashTable.c $(SRC_DIR)/hashTable.c $(SRC_DIR)/autoGrad.c

# Uncomment and adjust if test_loadData is needed
# test_loadData: $(TEST_DIR)/test_loadData.c $(SRC_DIR)/loadData.c
# 	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_loadData $(TEST_DIR)/test_loadData.c $(SRC_DIR)/loadData.c

clean:
	rm -f $(BIN_DIR)/*
