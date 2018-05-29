all: main copy

main: 
	g++ -std=c++11 -Wall -g -I../.. main.cpp -L../../build/lib -lcm -o main

copy:
	cp ../../build/lib/libcm.so ./libcm.so

clean:
	rm ./main ./libcm.so