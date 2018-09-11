#include "gtest/gtest.h"
#include "mcm/graph/dijkstra.hpp"
#include "mcm/pass/nce1/mode_selection.hpp"

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

TEST (dijkstra, mode_test)
{
    ModeSelectionNode source;
    source.remaining_output_channels = 192;

    source.parameters.input_height = 224;
    source.parameters.input_width = 224;
    source.parameters.output_height = 224;
    source.parameters.output_width = 224;
    source.parameters.input_channels = 3;
    source.parameters.output_channels = 64;
    source.parameters.kernel_x = 7;
    source.parameters.kernel_y = 7;
    source.parameters.stride_x = 2;
    source.parameters.stride_y = 2;

    ModeSelectionNode target;
    target.remaining_output_channels = 0;

    mv::DijkstraReturnValue<ModeSelectionNode, ModeSelectionDistance> shortestPath = mv::dijkstraRT<ModeSelectionNode, ModeSelectionDistance>(source, target, generateNeighboursComingFromValidModes, computeModeCost);
    std::cout << "Finished!" << std::endl;
}
