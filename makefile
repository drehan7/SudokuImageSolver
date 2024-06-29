CC=g++
CFLAGS = -Wall -Wpedantic -ggdb -Wextra -ulimit
OBJFILES = solver.o
TARGET = solver
TESTS = test

$(TESTS): tests/sudokutxt/tests.cpp objs/solver.o
	@$(CC) $(CFLAGS) tests/sudokutxt/tests.cpp objs/solver.o -o $(TESTS)
	@echo "\033[0;32mTests all good\033[0m"

objs/solver.o: src/solver.h src/solver.cpp
	@$(CC) $(CFLAGS) -c src/solver.cpp -o objs/solver.o

build:
	make
clean:
	rm objs/*.o
	rm test
  
