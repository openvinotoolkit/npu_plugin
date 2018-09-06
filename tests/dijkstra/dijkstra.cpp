#include "gtest/gtest.h"
#include "mcm/graph/dijkstra.hpp"

TEST (dijkstra, simple_test)
{
    int source = 0;
    int target = 5;

    std::function<std::vector<int>(int)> generateNeighbours = [](int a)
    {
        std::vector<int> toReturn;
        if(a > 3)
            return toReturn;
        toReturn.push_back(a+1);
        toReturn.push_back(a+2);
        return toReturn;
    };

    std::function<int(int, int)> computeCost = [](int u, int v)
    {
        if(u >= v)
            return u - v;
        else
            return v - u;
    };

    mv::DijkstraReturnValue<int, int> shortestPath = mv::dijkstraRT<int>(source, target, generateNeighbours, computeCost);
    std::cout << "Finished!" << std::endl;
}

