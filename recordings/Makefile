all: main copy

main: 
	g++ -std=c++11 -Wall -g -I../ recorded.cpp recorded_weights.cpp -L../build/lib -lcm -o recorded

copy:
	cp ../build/lib/libcm.so ./libcm.so

clean:
	rm ./recorded ./libcm.so ./*.dot ./recorded.blob ./recorded.json ./*.svg ./*.cpp
