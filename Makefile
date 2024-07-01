CXX		  := g++
CXX_FLAGS := -Wall -Wextra -std=c++2a -ggdb -fopenmp

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

debug: CXX_FLAGS += -DDEBUG -g
debug: clean $(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/main.cpp
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES) -lm -lpthread


clean:
	-rm $(BIN)/*
