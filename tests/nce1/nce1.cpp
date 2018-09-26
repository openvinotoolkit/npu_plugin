#include "gtest/gtest.h"
#include "mcm/algorithms/dijkstra.hpp"
#include "mcm/computation/resource/nce1.hpp"
#include "tests/include/MCMtest.hpp"


/*disabling test until mvNCCompile is working on master*/
TEST (nce1, DISABLED_HWconv_op_parameters)
{

   int no_index_lo = 72 ;
   int no_index_hi = 72 ;

   for( int no_index = no_index_lo; no_index <= no_index_hi; no_index = no_index + 1 )
   {
       MCMtest test("HWConv2D") ;


       test.addParam("input_tensor_shape","ih","224");
       test.addParam("input_tensor_shape","iw","224");
       test.addParam("input_tensor_shape","ic","3");
       test.addParam("input_tensor_shape","ib","1");
       test.addParam("convolution_operation","kh","3");
       test.addParam("convolution_operation","kw","3");
       test.addParam("convolution_operation","kf","3");
       test.addParam("convolution_operation","ph","0");
       test.addParam("convolution_operation","pw","0");
       test.addParam("convolution_operation","sh","2");
       test.addParam("convolution_operation","sw","2");
       test.addParam("convolution_operation","no", std::to_string(no_index));
       test.addParam("convolution_operation","wf","xavier");
       test.addParam("convolution_operation","ws","0.1");
       test.addParam("convolution_operation","bt","constant");
       test.addParam("convolution_operation","bv","2");

       test.generatePrototxt();

       std::string command1 = "$MDK_HOME/projects/Fathom/src2/mvNCCompile.py ./test.prototxt --new-parser --cpp";
       EXPECT_EQ (0, system(command1.c_str())) << "ERROR: non 0 return from compile";

       std::string command2 = "$MCM_HOME/python/tools/mcmCheck.sh -b ./cpp.blob -e ./Fathom_expected.npy -i ./test.png";
       EXPECT_EQ (0, system(command2.c_str())) << "ERROR: non 0 return from mcmCheck";

       test.saveResult();

    }
}





//Same tests present in SOH.py
TEST (nce1, split_over_h_1)
{
    mv::Nce1 nce;

    int max_output_lines = 30;

    mv::ConvolutionParameters param;

    param.input_height = 112;
    param.input_width = 112;
    param.kernel_x = 3;
    param.kernel_y = 3;
    param.stride_x = 2;
    param.stride_y = 2;
    param.pad_x_down = 1;
    param.pad_x_up = 1;
    param.pad_y_left = 1;
    param.pad_y_right = 1;
    param.output_width = 56;
    param.output_height = 56;
    param.input_channels = 3;
    param.output_channels = 64;

    std::vector<mv::SplitOverHSolution> result = nce.computeSplitsOverH(param, max_output_lines);
    //As usual, magic numbers obtained with the python compiler
    ASSERT_EQ(result.size(), 2);

    ASSERT_EQ(result[0].input_lines_processed, 60);
    ASSERT_EQ(result[0].output_lines_processed, 30);
    ASSERT_EQ(result[0].junk_output_before, 0);
    ASSERT_EQ(result[0].junk_output_after, 0);
    ASSERT_EQ(result[0].start_input_line, 0);
    ASSERT_EQ(result[0].end_input_line, 60);
    ASSERT_EQ(result[0].start_output_line, 0);
    ASSERT_EQ(result[0].end_output_line, 30);

    ASSERT_EQ(result[1].input_lines_processed, 54);
    ASSERT_EQ(result[1].output_lines_processed, 27);
    ASSERT_EQ(result[1].junk_output_before, 1);
    ASSERT_EQ(result[1].junk_output_after, 0);
    ASSERT_EQ(result[1].start_input_line, 58);
    ASSERT_EQ(result[1].end_input_line, 112);
    ASSERT_EQ(result[1].start_output_line, 30);
    ASSERT_EQ(result[1].end_output_line, 56);

    std::cout << "Finished!" << std::endl;
}

TEST (nce1, split_over_h_2)
{
    mv::Nce1 nce;
    mv::ConvolutionParameters param;

    param.input_height = 224;
    param.input_width = 224;
    param.kernel_x = 7;
    param.kernel_y = 7;
    param.stride_x = 2;
    param.stride_y = 2;
    param.pad_x_down = 3;
    param.pad_x_up = 3;
    param.pad_y_left = 3;
    param.pad_y_right = 3;
    param.output_width = 112;
    param.output_height = 112;
    param.input_channels = 3;
    param.output_channels = 64;

    unsigned max_output_lines = 9;

    std::vector<mv::SplitOverHSolution> result = nce.computeSplitsOverH(param, max_output_lines);

    //As usual, magic numbers obtained with the python compiler
    ASSERT_EQ(result.size(), 19);

    ASSERT_EQ(result[0].input_lines_processed, 18);
    ASSERT_EQ(result[0].output_lines_processed, 9);
    ASSERT_EQ(result[0].junk_output_before, 0);
    ASSERT_EQ(result[0].junk_output_after, 1);
    ASSERT_EQ(result[0].start_input_line, 0);
    ASSERT_EQ(result[0].end_input_line, 18);
    ASSERT_EQ(result[0].start_output_line, 0);
    ASSERT_EQ(result[0].end_output_line, 8);

    ASSERT_EQ(result[1].input_lines_processed, 18);
    ASSERT_EQ(result[1].output_lines_processed, 9);
    ASSERT_EQ(result[1].junk_output_before, 2);
    ASSERT_EQ(result[1].junk_output_after, 1);
    ASSERT_EQ(result[1].start_input_line, 12);
    ASSERT_EQ(result[1].end_input_line, 30);
    ASSERT_EQ(result[1].start_output_line, 8);
    ASSERT_EQ(result[1].end_output_line, 14);

    ASSERT_EQ(result[2].input_lines_processed, 18);
    ASSERT_EQ(result[2].output_lines_processed, 9);
    ASSERT_EQ(result[2].junk_output_before, 2);
    ASSERT_EQ(result[2].junk_output_after, 1);
    ASSERT_EQ(result[2].start_input_line, 24);
    ASSERT_EQ(result[2].end_input_line, 42);
    ASSERT_EQ(result[2].start_output_line, 14);
    ASSERT_EQ(result[2].end_output_line, 20);

    ASSERT_EQ(result[3].input_lines_processed, 18);
    ASSERT_EQ(result[3].output_lines_processed, 9);
    ASSERT_EQ(result[3].junk_output_before, 2);
    ASSERT_EQ(result[3].junk_output_after, 1);
    ASSERT_EQ(result[3].start_input_line, 36);
    ASSERT_EQ(result[3].end_input_line, 54);
    ASSERT_EQ(result[3].start_output_line, 20);
    ASSERT_EQ(result[3].end_output_line, 26);

    ASSERT_EQ(result[4].input_lines_processed, 18);
    ASSERT_EQ(result[4].output_lines_processed, 9);
    ASSERT_EQ(result[4].junk_output_before, 2);
    ASSERT_EQ(result[4].junk_output_after, 1);
    ASSERT_EQ(result[4].start_input_line, 48);
    ASSERT_EQ(result[4].end_input_line, 66);
    ASSERT_EQ(result[4].start_output_line, 26);
    ASSERT_EQ(result[4].end_output_line, 32);

    ASSERT_EQ(result[5].input_lines_processed, 18);
    ASSERT_EQ(result[5].output_lines_processed, 9);
    ASSERT_EQ(result[5].junk_output_before, 2);
    ASSERT_EQ(result[5].junk_output_after, 1);
    ASSERT_EQ(result[5].start_input_line, 60);
    ASSERT_EQ(result[5].end_input_line, 78);
    ASSERT_EQ(result[5].start_output_line, 32);
    ASSERT_EQ(result[5].end_output_line, 38);

    ASSERT_EQ(result[6].input_lines_processed, 18);
    ASSERT_EQ(result[6].output_lines_processed, 9);
    ASSERT_EQ(result[6].junk_output_before, 2);
    ASSERT_EQ(result[6].junk_output_after, 1);
    ASSERT_EQ(result[6].start_input_line, 72);
    ASSERT_EQ(result[6].end_input_line, 90);
    ASSERT_EQ(result[6].start_output_line, 38);
    ASSERT_EQ(result[6].end_output_line, 44);

    ASSERT_EQ(result[7].input_lines_processed, 18);
    ASSERT_EQ(result[7].output_lines_processed, 9);
    ASSERT_EQ(result[7].junk_output_before, 2);
    ASSERT_EQ(result[7].junk_output_after, 1);
    ASSERT_EQ(result[7].start_input_line, 84);
    ASSERT_EQ(result[7].end_input_line, 102);
    ASSERT_EQ(result[7].start_output_line, 44);
    ASSERT_EQ(result[7].end_output_line, 50);

    ASSERT_EQ(result[8].input_lines_processed, 18);
    ASSERT_EQ(result[8].output_lines_processed, 9);
    ASSERT_EQ(result[8].junk_output_before, 2);
    ASSERT_EQ(result[8].junk_output_after, 1);
    ASSERT_EQ(result[8].start_input_line, 96);
    ASSERT_EQ(result[8].end_input_line, 114);
    ASSERT_EQ(result[8].start_output_line, 50);
    ASSERT_EQ(result[8].end_output_line, 56);

    ASSERT_EQ(result[9].input_lines_processed, 18);
    ASSERT_EQ(result[9].output_lines_processed, 9);
    ASSERT_EQ(result[9].junk_output_before, 2);
    ASSERT_EQ(result[9].junk_output_after, 1);
    ASSERT_EQ(result[9].start_input_line, 108);
    ASSERT_EQ(result[9].end_input_line, 126);
    ASSERT_EQ(result[9].start_output_line, 56);
    ASSERT_EQ(result[9].end_output_line, 62);

    ASSERT_EQ(result[10].input_lines_processed, 18);
    ASSERT_EQ(result[10].output_lines_processed, 9);
    ASSERT_EQ(result[10].junk_output_before, 2);
    ASSERT_EQ(result[10].junk_output_after, 1);
    ASSERT_EQ(result[10].start_input_line, 120);
    ASSERT_EQ(result[10].end_input_line, 138);
    ASSERT_EQ(result[10].start_output_line, 62);
    ASSERT_EQ(result[10].end_output_line, 68);

    ASSERT_EQ(result[11].input_lines_processed, 18);
    ASSERT_EQ(result[11].output_lines_processed, 9);
    ASSERT_EQ(result[11].junk_output_before, 2);
    ASSERT_EQ(result[11].junk_output_after, 1);
    ASSERT_EQ(result[11].start_input_line, 132);
    ASSERT_EQ(result[11].end_input_line, 150);
    ASSERT_EQ(result[11].start_output_line, 68);
    ASSERT_EQ(result[11].end_output_line, 74);

    ASSERT_EQ(result[12].input_lines_processed, 18);
    ASSERT_EQ(result[12].output_lines_processed, 9);
    ASSERT_EQ(result[12].junk_output_before, 2);
    ASSERT_EQ(result[12].junk_output_after, 1);
    ASSERT_EQ(result[12].start_input_line, 144);
    ASSERT_EQ(result[12].end_input_line, 162);
    ASSERT_EQ(result[12].start_output_line, 74);
    ASSERT_EQ(result[12].end_output_line, 80);

    ASSERT_EQ(result[13].input_lines_processed, 18);
    ASSERT_EQ(result[13].output_lines_processed, 9);
    ASSERT_EQ(result[13].junk_output_before, 2);
    ASSERT_EQ(result[13].junk_output_after, 1);
    ASSERT_EQ(result[13].start_input_line, 156);
    ASSERT_EQ(result[13].end_input_line, 174);
    ASSERT_EQ(result[13].start_output_line, 80);
    ASSERT_EQ(result[13].end_output_line, 86);

    ASSERT_EQ(result[14].input_lines_processed, 18);
    ASSERT_EQ(result[14].output_lines_processed, 9);
    ASSERT_EQ(result[14].junk_output_before, 2);
    ASSERT_EQ(result[14].junk_output_after, 1);
    ASSERT_EQ(result[14].start_input_line, 168);
    ASSERT_EQ(result[14].end_input_line, 186);
    ASSERT_EQ(result[14].start_output_line, 86);
    ASSERT_EQ(result[14].end_output_line, 92);

    ASSERT_EQ(result[15].input_lines_processed, 18);
    ASSERT_EQ(result[15].output_lines_processed, 9);
    ASSERT_EQ(result[15].junk_output_before, 2);
    ASSERT_EQ(result[15].junk_output_after, 1);
    ASSERT_EQ(result[15].start_input_line, 180);
    ASSERT_EQ(result[15].end_input_line, 198);
    ASSERT_EQ(result[15].start_output_line, 92);
    ASSERT_EQ(result[15].end_output_line, 98);

    ASSERT_EQ(result[16].input_lines_processed, 18);
    ASSERT_EQ(result[16].output_lines_processed, 9);
    ASSERT_EQ(result[16].junk_output_before, 2);
    ASSERT_EQ(result[16].junk_output_after, 1);
    ASSERT_EQ(result[16].start_input_line, 192);
    ASSERT_EQ(result[16].end_input_line, 210);
    ASSERT_EQ(result[16].start_output_line, 98);
    ASSERT_EQ(result[16].end_output_line, 104);

    ASSERT_EQ(result[17].input_lines_processed, 18);
    ASSERT_EQ(result[17].output_lines_processed, 9);
    ASSERT_EQ(result[17].junk_output_before, 2);
    ASSERT_EQ(result[17].junk_output_after, 1);
    ASSERT_EQ(result[17].start_input_line, 204);
    ASSERT_EQ(result[17].end_input_line, 222);
    ASSERT_EQ(result[17].start_output_line, 104);
    ASSERT_EQ(result[17].end_output_line, 110);

    ASSERT_EQ(result[18].input_lines_processed, 8);
    ASSERT_EQ(result[18].output_lines_processed, 4);
    ASSERT_EQ(result[18].junk_output_before, 2);
    ASSERT_EQ(result[18].junk_output_after, 0);
    ASSERT_EQ(result[18].start_input_line, 216);
    ASSERT_EQ(result[18].end_input_line, 224);
    ASSERT_EQ(result[18].start_output_line, 110);
    ASSERT_EQ(result[18].end_output_line, 112);

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


