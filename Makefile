CC=gcc
CFLAGS=-I include
SRC_DIR=src
TEST_DIR=test
BIN_DIR=bin
LIB_SOURCES=$(wildcard $(SRC_DIR)/*.c)

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

all: test_autoGrad 

# Test targets
test_autoGrad: $(TEST_DIR)/test_autoGrad.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$(@F)
