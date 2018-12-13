SRC := $(wildcard *.fbs)
OBJS := $(patsubst %.fbs, compiledSchemas/%_generated.h, $(wildcard *.fbs))
FLATC := flatc

compiledSchemas/%_generated.h: %.fbs
	$(FLATC) -o compiledSchemas/ $< --cpp

all: clean $(OBJS)

clean:
	-rm compiledSchemas/*_generated.h




