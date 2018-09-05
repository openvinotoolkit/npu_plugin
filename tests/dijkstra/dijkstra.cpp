#include "gtest/gtest.h"
#include "mcm/graph/dijkstra.hpp"

TEST (dijkstra, simple_test)
{
    int source = 0;
    int target = 4;

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

    std::vector<int> shortestPath = mv::dijkstraRT<int>(source, target, generateNeighbours, computeCost);

}

TEST (dijkstra, complex_test)
{
    int source = 192;
    int target = 0;

    std::function<std::vector<int>(int)> generateNeighbours = [](int a)
    {
        int modes_output_channel = [256, 128, 64, 32, 16];
        std::vector<int> toReturn;
        for(int i = 0; i < 5; ++i)
        {
            int toAdd;
            int remaining_output_channels = a - modes_output_channel;
            toAdd = remaining_output_channels < 0 ? 0 : remaining_output_channels;
            toReturn.push_back(toAdd);
        }
        return toReturn;
    };

    std::function<int(int, int)> computeCost = [](int u, int v)
    {
        int modes_output_channel = [256, 128, 64, 32, 16];

        for(int i = 0; i < 5; ++i)
        {
            int toAdd;
            int remaining_output_channels = a - modes_output_channel;
            toAdd = remaining_output_channels < 0 ? 0 : remaining_output_channels;
        }
    };

    std::vector<int> shortestPath = mv::dijkstraRT<int>(source, target, generateNeighbours, computeCost);

}

