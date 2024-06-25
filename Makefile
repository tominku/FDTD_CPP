CXX		  := nvcc
#CXX_FLAGS := -Wall -Wextra -std=c++17 -ggdb -fopenmp
CXX_FLAGS := -std=c++17

BIN		:= bin
SRC		:= src
INCLUDE	:= include
LIB		:= lib

LIBRARIES	:=
EXECUTABLE	:= main


all: $(BIN)/$(EXECUTABLE)

run: clean all
	#clear
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/main.cu
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES) -lm

clean:
	-rm $(BIN)/*
