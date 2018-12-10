#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "graphfile_generated.h"
#include "deserialize.hpp"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/registry.h"
#include "flatbuffers/util.h"
#include <gtest/gtest.h>

//need to break up
// TEST(graphFile, header_SummaryHeader_version_Version)	
// {
//     Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
//     Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
//     const auto graph1 = GetGraphFile(blob_1.get_ptr());
//     const auto graph2 = GetGraphFile(blob_2.get_ptr());
    
//     EXPECT_EQ(graph1->header()->version()->majorV(),graph2->header()->version()->majorV());
//     EXPECT_EQ(graph1->header()->version()->minorV(),graph2->header()->version()->minorV());
//     EXPECT_EQ(graph1->header()->version()->patchV(),graph2->header()->version()->patchV());
//     EXPECT_EQ(graph1->header()->version()->hash()->c_str(),graph2->header()->version()->hash()->c_str());
// }

TEST(graphFile, header_SummaryHeader_task_count)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is task_cout present in graph1
    auto taskCountPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_TASK_COUNT);

    if(taskCountPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_TASK_COUNT)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->task_count(),graph2->header()->task_count()); //Test if equal
    }
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_layer_count)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is layer_cout present in graph1
    auto layerCountPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_LAYER_COUNT);
    
    if(layerCountPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_LAYER_COUNT)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->layer_count(),graph2->header()->layer_count()); //Then test if equal
    }
    else
        SUCCEED();
}

//Need to update to check if TensorReference table (contains dimension field) is present
TEST(graphFile, header_SummaryHeader_net_input_TensorReference_dimensions)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_input present in graph1
    auto netInputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_INPUT);

     if(netInputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_INPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_input()->size(); i++) { //No. of TensorReference tables

            for (flatbuffers::uoffset_t j = 0; j < graph1->header()->net_input()->Get(i)->dimensions()->size(); j++) { //For each dimension check if equal
            
                EXPECT_EQ(graph1->header()->net_input()->Get(i)->dimensions()->Get(j),graph2->header()->net_input()->Get(i)->dimensions()->Get(j));
            }
        }  
    } 
    else
        SUCCEED();
}

//Need to update to check if TensorReference table (contains dimension field) is present
TEST(graphFile, header_SummaryHeader_net_output_TensorReference_dimensions)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netOutputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

     if(netOutputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_output()->size(); i++) { //No. of TensorReference tables

            for (flatbuffers::uoffset_t j = 0; j < graph1->header()->net_output()->Get(i)->dimensions()->size(); j++) { //For each dimension check if equal
            
                EXPECT_EQ(graph1->header()->net_output()->Get(i)->dimensions()->Get(j),graph2->header()->net_output()->Get(i)->dimensions()->Get(j));
            }
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_resources_Resources_shave_mask)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is shave_mask present in graph1
    auto shaveMaskPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->resources(), Resources::VT_SHAVE_MASK);
    
    if(shaveMaskPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->resources(), Resources::VT_SHAVE_MASK)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->resources()->shave_mask(),graph2->header()->resources()->shave_mask()); //Then test if equal
    }
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_resources_Resources_nce1_mask)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is nce1_mask present in graph1
    auto nce1MaskPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->resources(), Resources::VT_NCE1_MASK);
    
    if(nce1MaskPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->resources(),Resources::VT_NCE1_MASK)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->resources()->nce1_mask(),graph2->header()->resources()->nce1_mask()); //Then test if equal
    }
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_resources_Resources_dpu_mask)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is dpu_mask present in graph1
    auto dpuMaskPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->resources(), Resources::VT_DPU_MASK);
    
    if(dpuMaskPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->resources(),Resources::VT_DPU_MASK)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->resources()->dpu_mask(),graph2->header()->resources()->dpu_mask()); //Then test if equal
    }
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_resources_Resources_leon_cmx)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is leon_cmx present in graph1
    auto leonCmxPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->resources(), Resources::VT_LEON_CMX);
    
    if(leonCmxPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->resources(),Resources::VT_LEON_CMX)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->resources()->leon_cmx(),graph2->header()->resources()->leon_cmx()); //Then test if equal
    }
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_resources_Resources_nn_cmx)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is nn_cmx present in graph1
    auto nnCmxPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->resources(), Resources::VT_NN_CMX);
    
    if(nnCmxPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->resources(),Resources::VT_NN_CMX)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->resources()->nn_cmx(),graph2->header()->resources()->nn_cmx()); //Then test if equal
    }
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_resources_Resources_ddr_scratch)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is ddr_scratch present in graph1
    auto ddrScratchPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->resources(), Resources::VT_NN_CMX);
    
    if(ddrScratchPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->resources(),Resources::VT_DDR_SCRATCH)); //Check if present in graph2
        EXPECT_EQ(graph1->header()->resources()->ddr_scratch(),graph2->header()->resources()->ddr_scratch()); //Then test if equal
    }
    else
        SUCCEED();
}


//Not written in POC blob
TEST(DISABLE_graphFile, task_lists_TaskList_content_Task_nodeID)		
{
   SUCCEED();
}

//Need to check if Task table is present (contains sourceTaskIDs field)
TEST(graphFile, task_lists_TaskList_content_Task_sourceTaskIDs)	
{
    Blob blob_1("/home/john/blobv3/try4.blob");
    Blob blob_2("/home/john/blobv3/try5.blob");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    auto TaskLists_graph1 = graph1->task_lists(); 
    auto TaskLists_graph2 = graph2->task_lists();

    for (flatbuffers::uoffset_t j = 0; j < TaskLists_graph1->size(); j++) { //No. of TaskList tables
        
        auto content_size = TaskLists_graph1->Get(j)->content()->size(); //No. of Task tables

        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) {

            auto sourceTaskIDs_size = TaskLists_graph1->Get(j)->content()->Get(i)->sourceTaskIDs()->size();

            for (flatbuffers::uoffset_t k = 0; k < sourceTaskIDs_size; k++) {

                auto sourceTaskIDs_graph1 = TaskLists_graph1->Get(j)->content()->Get(i)->sourceTaskIDs()->Get(k);
                auto sourceTaskIDs_graph2 = TaskLists_graph2->Get(j)->content()->Get(i)->sourceTaskIDs()->Get(k);
      
                EXPECT_EQ(sourceTaskIDs_graph1, sourceTaskIDs_graph2);
            }
        }
    }
}

//Add ASSERT
TEST(graphFile, task_lists_TaskList_content_Task_associated_barriers_BarrierReference_wait_barriers)	
{
    Blob blob_1("/home/john/blobv3/try4.blob");
    Blob blob_2("/home/john/blobv3/try5.blob");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    auto TaskLists_graph1 = graph1->task_lists(); 
    auto TaskLists_graph2 = graph2->task_lists();

    for (flatbuffers::uoffset_t j = 0; j < TaskLists_graph1->size(); j++) { //No. of TaskList tables
        
        auto content_size = TaskLists_graph1->Get(j)->content()->size(); //No. of Task tables
       
        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) { //For each Task Table

            auto wait_barrier_graph1 = TaskLists_graph1->Get(j)->content()->Get(i)->associated_barriers()->wait_barrier(); //get wait_barrier
            auto wait_barrier_graph2 = TaskLists_graph2->Get(j)->content()->Get(i)->associated_barriers()->wait_barrier(); //get wait_barrier
      
            EXPECT_EQ(wait_barrier_graph1,wait_barrier_graph2);
        }
    }
}

//Add ASSERT
TEST(graphFile, task_lists_TaskList_content_Task_associated_barriers_BarrierReference_update_barriers)	
{
    Blob blob_1("/home/john/blobv3/try4.blob");
    Blob blob_2("/home/john/blobv3/try5.blob");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    auto task_lists_graph1 = graph1->task_lists(); 
    auto task_lists_graph2 = graph2->task_lists();

    //Is task_lists present in graph1
    auto taskListPresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_TASK_LISTS);

    //if present in graph1
    if(taskListPresentGraph1) {
        for (flatbuffers::uoffset_t j = 0; j < task_lists_graph1->size(); j++) { //No. of TaskList tables
        
            auto task_tables_size = task_lists_graph1->Get(j)->content()->size(); //No. of Task tables
       
            for (flatbuffers::uoffset_t i = 0; i < task_tables_size; i++) { //For each Task Table

                auto update_barriers_present = flatbuffers::IsFieldPresent(task_lists_graph1->Get(j)->content()->Get(i)->associated_barriers(),BarrierReference::VT_UPDATE_BARRIERS);

                if(update_barriers_present) { //If present
                    auto update_barriers_size = task_lists_graph1->Get(j)->content()->Get(i)->associated_barriers()->update_barriers()->size(); //Get Size of update_barriers vector
                    for (flatbuffers::uoffset_t k = 0; k < update_barriers_size ; k++) { //For each update_barriers element check if equal
            
                        auto update_barriers_graph1 = task_lists_graph1->Get(j)->content()->Get(i)->associated_barriers()->update_barriers()->Get(k); //get update_barriers
                        auto update_barriers_graph2 = task_lists_graph2->Get(j)->content()->Get(i)->associated_barriers()->update_barriers()->Get(k); //get update_barriers

                        EXPECT_EQ(update_barriers_graph1,update_barriers_graph2);
                    }
                }
                else
                    SUCCEED();
            }         
        }
    }
    else
        SUCCEED();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}