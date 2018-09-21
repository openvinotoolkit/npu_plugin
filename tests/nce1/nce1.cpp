#include "gtest/gtest.h"
#include "mcm/algorithms/dijkstra.hpp"
#include "mcm/computation/resource/nce1.hpp"

//Same test present in SOH.py
TEST (nce1, split_over_h_1)
{

    mv::Nce1 nce;
    mv::ConvolutionParameters parameters;

    int max_output_lines = 30;

    parameters.input_height = 112;
    parameters.output_height = 56;
    parameters.kernel_x = 3;
    parameters.kernel_y = 3;
    parameters.stride_x = 2;
    parameters.stride_y = 2;
    parameters.pad_x_up = 1;
    parameters.pad_x_down = 1;

    std::vector<mv::SplitOverHSolution> result = nce.computeSplitsOverH(parameters, max_output_lines);

    std::cout << "Finished!" << std::endl;
}

//This is a simple test case, no splits are involved, 1 step to solve it.
TEST (nce1, mode_selection_resnet_first_conv)
{

    mv::Nce1 nce;
    mv::ModeSelectionNode source;

    source.parameters.input_height = 224;
    source.parameters.input_width = 224;
    source.parameters.output_height = 112;
    source.parameters.output_width = 112;
    source.parameters.input_channels = 3;
    source.parameters.output_channels = 64;
    source.parameters.kernel_x = 7;
    source.parameters.kernel_y = 7;
    source.parameters.stride_x = 2;
    source.parameters.stride_y = 2;
    source.remaining_output_channels = source.parameters.output_channels;

    mv::DijkstraReturnValue<mv::ModeSelectionNode, mv::ModeSelectionDistance> shortestPath = nce.optimize_convolution(source);

    //ASSERTION Values given by python compiler
    //First: Check that paths have the same size
    ASSERT_EQ(shortestPath.distances.size(),1);

    //Second: If they have the same size, check that all modes corrispond (no permutation allowed)
    std::vector<unsigned> expected_modes = {2};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].mode, expected_modes[i]);

    //Third: If all the modes correspond, splits should be equal as well
    std::vector<int> expected_splits = {1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].num_splits, expected_splits[i]);

    //Fourth: If all the modes correspond, check that total solution cost correspnds
    unsigned total_cost = shortestPath.distances[shortestPath.distances.size()-1].cost;
    ASSERT_EQ(total_cost, 2458624);

    //Finished
    std::cout << "Finished!" << std::endl;
}

//This is a simple test case, no splits are involved, more steps to solve it.
TEST (nce1, mode_selection_resnet_another_conv)
{
    mv::Nce1 nce;
    mv::ModeSelectionNode source;

    source.parameters.input_height = 56;
    source.parameters.input_width = 56;
    source.parameters.output_height = 56;
    source.parameters.output_width = 56;
    source.parameters.input_channels = 64;
    source.parameters.output_channels = 224;
    source.parameters.kernel_x = 1;
    source.parameters.kernel_y = 1;
    source.parameters.stride_x = 1;
    source.parameters.stride_y = 1;
    source.remaining_output_channels = source.parameters.output_channels;

    mv::DijkstraReturnValue<mv::ModeSelectionNode, mv::ModeSelectionDistance> shortestPath = nce.optimize_convolution(source);

    //ASSERTION Values given by python compiler
    //First: Check that paths have the same size
    ASSERT_EQ(shortestPath.distances.size(),3);

    //Second: If they have the same size, check that all modes corrispond (no permutation allowed)
    std::vector<unsigned> expected_modes = {3, 2, 1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].mode, expected_modes[i]);

    //Third: If all the modes correspond, splits should be equal as well
    std::vector<int> expected_splits = {1, 1, 1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].num_splits, expected_splits[i]);

    //Fourth: If all the modes correspond, check that total solution cost correspnds
    unsigned total_cost = shortestPath.distances[shortestPath.distances.size()-1].cost;
    ASSERT_EQ(total_cost, 189616);

    //Finished
    std::cout << "Finished!" << std::endl;
}

//Splits are involved, more steps to solve it.
//TODO: find a proper convolution
TEST (nce1, split_convolution)
{
    std::cout << "Finished!" << std::endl;
}


