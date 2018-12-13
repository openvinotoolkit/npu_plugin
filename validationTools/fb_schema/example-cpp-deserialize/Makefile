EXEC = deserialize

WARNING_FLAGS 	:= -pedantic-errors -Wall -Wextra -Wno-format-security

COMPILATION_FLAGS := -std=c++17 -fPIC -fconcepts -c -g

INCLUDES := -I../flatbuffers/include -I. 

LIBS := -lstdc++ -lpthread

SRCS := $(wildcard *.cpp)

OBJS := $(patsubst %.cpp, %.o, $(wildcard *.cpp))

STANDALONE_HEADERS :=

LIB := graph.so

LIB_OBJS := $(filter-out main.o,$(OBJS))

LIB_FLAGS := -shared

# .PHONY: all clean

%.o: %.cpp
	gcc $^ $(INCLUDES) $(LIBS) $(COMPILATION_FLAGS) $(WARNING_FLAGS) -g \
		-o $@ $(STANDALONE_HEADERS) 

$(LIB): $(LIB_OBJS)
	gcc $(LIB_OBJS) $(LIBS) $(LIB_FLAGS) -g -o $@ 

$(EXEC): $(LIB) main.o
	# gcc main.o $(LIB) $(LIBS) -o $(EXEC)
	# sudo ldconfig

	gcc main.o -L . -l:graph.so /usr/lib/libgtest.a /usr/lib/libgtest_main.a -g $(LIBS) -o $(EXEC) 

	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(PWD); 

all: clean $(EXEC)

clean:
	-rm *.o $(EXEC)
