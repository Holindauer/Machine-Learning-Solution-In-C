CC=gcc
CFLAGS=-I include
SRC_DIR=src
TEST_DIR=test
BIN_DIR=bin

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

all: main test

main: $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/main $(SRC_DIR)/main.c

test: $(TEST_DIR)/test_main.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_main $(TEST_DIR)/test_main.c

clean:
	rm -f $(BIN_DIR)/*
