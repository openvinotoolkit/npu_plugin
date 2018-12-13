WARNING_FLAGS 	:= -pedantic-errors -Wall -Wextra -Werror

INCLUDES := -I../flatbuffers/include -I.

LIBS := -lstdc++

SRCS := $(wildcard *.cpp)

EXEC = serializeTest

OBJS := $(patsubst %.cpp, %.o, $(wildcard *.cpp))

STANDALONE_HEADERS := -include fakeGuest.hpp -include fakeHost.hpp

# .PHONY: all clean

%.o: %.cpp
	gcc -c $^ $(INCLUDES) $(LIBS) $(WARNING_FLAGS) -o $@ $(STANDALONE_HEADERS) -O3

$(EXEC): $(OBJS)
	gcc $(OBJS) -o $(EXEC) $(LIBS)
	./serializeTest

all: clean $(EXEC)

clean:
	-rm *.o serializeTest