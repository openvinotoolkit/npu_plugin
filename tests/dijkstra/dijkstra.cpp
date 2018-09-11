#include "gtest/gtest.h"
#include "mcm/graph/dijkstra.hpp"
#include "mcm/pass/nce1/mode_selection.hpp"

//This is a simple test case, no splits are involved.
TEST (dijkstra, resnet_first_conv_no_splits)
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

//This is a simple test case, no splits are involved.
TEST (dijkstra, resnet_first_conv)
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

