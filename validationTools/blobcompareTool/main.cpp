#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "../../KeemBayFBSchema/compiledSchemas/graphfile_generated.h"
#include "deserialize.hpp"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/registry.h"
#include "flatbuffers/util.h"
#include <gtest/gtest.h>
#include "testEnvironment.hpp"

//need to break up
TEST(graphFile, header_SummaryHeader_version_Version)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    
    EXPECT_EQ(graph1->header()->version()->majorV(),graph2->header()->version()->majorV());
    EXPECT_EQ(graph1->header()->version()->minorV(),graph2->header()->version()->minorV());
    EXPECT_EQ(graph1->header()->version()->patchV(),graph2->header()->version()->patchV());
}

TEST(graphFile, header_SummaryHeader_net_input_TensorReference_dimensions)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_input present in graph1
    auto netInputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_INPUT);

    if(netInputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_INPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_input()->size(); i++) { //No. of TensorReference tables
            
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_input()->Get(i), TensorReference::VT_DIMENSIONS)); //check dimensions field present

            auto dimensionsPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->net_input()->Get(i), TensorReference::VT_DIMENSIONS);  //are dimensions present

            if(dimensionsPresentGraph1) {
                for (flatbuffers::uoffset_t j = 0; j < graph1->header()->net_input()->Get(i)->dimensions()->size(); j++) { //For each dimension check if equal
                    
                    ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_input()->Get(i), TensorReference::VT_DIMENSIONS));
                    EXPECT_EQ(graph1->header()->net_input()->Get(i)->dimensions()->Get(j),graph2->header()->net_input()->Get(i)->dimensions()->Get(j));
                }
            }
            else
                SUCCEED();
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_net_input_TensorReference_strides)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_input present in graph1
    auto netInputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_INPUT);

     if(netInputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_INPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_input()->size(); i++) { //No. of TensorReference tables

            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_input()->Get(i), TensorReference::VT_STRIDES));

            auto stridesPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->net_input()->Get(i), TensorReference::VT_STRIDES);

            if(stridesPresentGraph1) {
                //For each dimension check if equal
                for (flatbuffers::uoffset_t j = 0; j < graph1->header()->net_input()->Get(i)->strides()->size(); j++) { 
                    
                    ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_input()->Get(i), TensorReference::VT_STRIDES));
                    EXPECT_EQ(graph1->header()->net_input()->Get(i)->strides()->Get(j),graph2->header()->net_input()->Get(i)->strides()->Get(j));
                }
            }
            else
                SUCCEED();
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_net_output_TensorReference_data_IndirectDataReference_data_index)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netOutputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

     if(netOutputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_output()->size(); i++) { //No. of TensorReference tables
            
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_output()->Get(i)->data(),IndirectDataReference::VT_DATA_INDEX)); //check data_index present
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_output()->Get(i)->data(),IndirectDataReference::VT_DATA_INDEX)); //check data_index present
            
            EXPECT_EQ(graph1->header()->net_output()->Get(i)->data()->data_index(),graph2->header()->net_output()->Get(i)->data()->data_index());
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_net_input_TensorReference_locale)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netinputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

    if(netinputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_input()->size(); i++) { //No. of TensorReference tables

            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_input()->Get(i),TensorReference::VT_LOCALE)); //check locale present
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_input()->Get(i),TensorReference::VT_LOCALE)); //check locale present

            EXPECT_EQ(graph1->header()->net_input()->Get(i)->locale(),graph2->header()->net_input()->Get(i)->locale());
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_net_input_TensorReference_data_dtype)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netInputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

     if(netInputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_input()->size(); i++) { //No. of TensorReference tables
      
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_input()->Get(i),TensorReference::VT_DATA_DTYPE)); //check locale present
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_input()->Get(i),TensorReference::VT_DATA_DTYPE)); //check locale present

            EXPECT_EQ(graph1->header()->net_input()->Get(i)->data_dtype(),graph2->header()->net_input()->Get(i)->data_dtype());
        }  
    } 
    else
        SUCCEED();
}


TEST(graphFile, header_SummaryHeader_net_output_TensorReference_dimensions)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netOutputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

    if(netOutputPresentGraph1) {

        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_output()->size(); i++) { //No. of TensorReference tables

             ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_output()->Get(i), TensorReference::VT_DIMENSIONS)); //check dimensions field present
            auto dimensionsPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->net_output()->Get(i), TensorReference::VT_DIMENSIONS);

            if(dimensionsPresentGraph1) {
                
                //For each dimension check if equal
                for (flatbuffers::uoffset_t j = 0; j < graph1->header()->net_output()->Get(i)->dimensions()->size(); j++) { 
                    
                    ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_output()->Get(i), TensorReference::VT_DIMENSIONS));

                    EXPECT_EQ(graph1->header()->net_output()->Get(i)->dimensions()->Get(j),graph2->header()->net_output()->Get(i)->dimensions()->Get(j));
                }
            }
            else
                SUCCEED();
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_net_output_TensorReference_strides)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netOutputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

     if(netOutputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_output()->size(); i++) { //No. of TensorReference tables

            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_output()->Get(i), TensorReference::VT_STRIDES));

            auto stridesPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header()->net_output()->Get(i), TensorReference::VT_STRIDES);

            if(stridesPresentGraph1) {
                
                for (flatbuffers::uoffset_t j = 0; j < graph1->header()->net_output()->Get(i)->strides()->size(); j++) { //For each dimension check if equal
            
                    ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_output()->Get(i), TensorReference::VT_STRIDES));
                    EXPECT_EQ(graph1->header()->net_output()->Get(i)->strides()->Get(j),graph2->header()->net_output()->Get(i)->strides()->Get(j));
                }
            }
            else
                SUCCEED();
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_net_output_TensorReference_locale)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netOutputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

     if(netOutputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_output()->size(); i++) { //No. of TensorReference tables
      
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_output()->Get(i),TensorReference::VT_LOCALE)); //check locale present
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_output()->Get(i),TensorReference::VT_LOCALE)); //check locale present

            EXPECT_EQ(graph1->header()->net_output()->Get(i)->locale(),graph2->header()->net_output()->Get(i)->locale());
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_net_output_TensorReference_data_dtype)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is net_output present in graph1
    auto netOutputPresentGraph1 = flatbuffers::IsFieldPresent(graph1->header(), SummaryHeader::VT_NET_OUTPUT);

    if(netOutputPresentGraph1) {
        ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header(), SummaryHeader::VT_NET_OUTPUT)); //Check if present in graph2

        for (flatbuffers::uoffset_t i = 0; i < graph1->header()->net_output()->size(); i++) { //No. of TensorReference tables
      
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->header()->net_output()->Get(i),TensorReference::VT_DATA_DTYPE)); //check locale present
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->header()->net_output()->Get(i),TensorReference::VT_DATA_DTYPE)); //check locale present

            EXPECT_EQ(graph1->header()->net_output()->Get(i)->data_dtype(),graph2->header()->net_output()->Get(i)->data_dtype());
        }  
    } 
    else
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_resources_Resources_shave_mask)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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

//Insert orignal structure - Links table here

//This test needs ASSERTS()
//Leaving for now as not in blob
TEST(graphFile, task_lists_TaskList_content_Task_nodeID)		
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is task_lists present in graph1
    auto taskListsPresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_TASK_LISTS);

    if(taskListsPresentGraph1) { //if present

    for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
        auto content_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables

        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) {
                
                //is nodeIDPresentGraph1 present
                auto nodeIDPresentGraph1 = flatbuffers::IsFieldPresent(graph1->task_lists()->Get(j)->content()->Get(i), Task::VT_NODEID);

                if(nodeIDPresentGraph1) {
                    auto nodeID_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->nodeID();
                    auto nodeID_graph2 = graph2->task_lists()->Get(j)->content()->Get(i)->nodeID();
                    
                    EXPECT_EQ(nodeID_graph1, nodeID_graph2);
                }
                else 
                    SUCCEED();
        }
    }
    }
    else 
        SUCCEED();
}

TEST(graphFile, barrier_table_Barrier_barrier_id)		
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is barrier_table present in graph1
    auto barrierTablePresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_BARRIER_TABLE);

    if(barrierTablePresentGraph1) { //if present

        for (flatbuffers::uoffset_t j = 0; j < graph1->barrier_table()->size(); j++) { //No. of Barrier tables
    
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->barrier_table()->Get(j), Barrier::VT_BARRIER_ID)); //is barrier id present in graph1

            auto barrierIdPresentGraph1 = flatbuffers::IsFieldPresent(graph1->barrier_table()->Get(j), Barrier::VT_BARRIER_ID);

            if(barrierIdPresentGraph1) {

                ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->barrier_table()->Get(j), Barrier::VT_BARRIER_ID)); //is it present in graph 2
            
                auto barrier_id_graph1 = graph1->barrier_table()->Get(j)->barrier_id();
                auto barrier_id_graph2 = graph2->barrier_table()->Get(j)->barrier_id();
                    
                EXPECT_EQ(barrier_id_graph1, barrier_id_graph2);
            }
            else 
                SUCCEED();
        }
    }
    else 
        SUCCEED();
}

TEST(graphFile, barrier_table_Barrier_consumer_count)		
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is barrier_table present in graph1
    auto barrierTablePresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_BARRIER_TABLE);

    if(barrierTablePresentGraph1) { //if present

        for (flatbuffers::uoffset_t j = 0; j < graph1->barrier_table()->size(); j++) { //No. of Barrier tables
    
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->barrier_table()->Get(j), Barrier::VT_CONSUMER_COUNT)); //is consumer count present in graph1

            //is consumer_count present in graph1
            auto consumerCountPresentGraph1 = flatbuffers::IsFieldPresent(graph1->barrier_table()->Get(j), Barrier::VT_CONSUMER_COUNT);

            if(consumerCountPresentGraph1) {
            
                ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->barrier_table()->Get(j), Barrier::VT_CONSUMER_COUNT)); //is consumer count present in graph2

                auto consumer_count_graph1 = graph1->barrier_table()->Get(j)->consumer_count();
                auto consumer_count_graph2 = graph2->barrier_table()->Get(j)->consumer_count();
                    
                EXPECT_EQ(consumer_count_graph1, consumer_count_graph2);
            }
            else 
                SUCCEED();
        }
    }
    else 
        SUCCEED();
}

TEST(graphFile, barrier_table_Barrier_producer_count)		
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is barrier_table present in graph1
    auto barrierTablePresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_BARRIER_TABLE);

    if(barrierTablePresentGraph1) { //if present

        for (flatbuffers::uoffset_t j = 0; j < graph1->barrier_table()->size(); j++) { //No. of Barrier tables

            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->barrier_table()->Get(j), Barrier::VT_PRODUCER_COUNT)); //is producer count present in graph1
    
            //is producer_count present in graph1
            auto producerCountPresentGraph1 = flatbuffers::IsFieldPresent(graph1->barrier_table()->Get(j), Barrier::VT_PRODUCER_COUNT);

            if(producerCountPresentGraph1) {
            
                ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->barrier_table()->Get(j), Barrier::VT_PRODUCER_COUNT)); //is producer count present in graph2

                auto producer_count_graph1 = graph1->barrier_table()->Get(j)->producer_count();
                auto producer_count_graph2 = graph2->barrier_table()->Get(j)->producer_count();
                    
                EXPECT_EQ(producer_count_graph1, producer_count_graph2);
            }
            else 
                SUCCEED();
        }
    }
    else 
        SUCCEED();
}

//Need to check if Task table is present (contains sourceTaskIDs field)
TEST(graphFile, task_lists_TaskList_content_Task_sourceTaskIDs)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
        auto content_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables

        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) {

            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->task_lists()->Get(j)->content()->Get(i), Task::VT_SOURCETASKIDS)); //is source task ids present graph1

            auto sourceTaskIDs_size = graph1->task_lists()->Get(j)->content()->Get(i)->sourceTaskIDs()->size();

            for (flatbuffers::uoffset_t k = 0; k < sourceTaskIDs_size; k++) {

                ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->task_lists()->Get(j)->content()->Get(i), Task::VT_SOURCETASKIDS)); //is source task ids present graph2

                auto sourceTaskIDs_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->sourceTaskIDs()->Get(k);
                auto sourceTaskIDs_graph2 = graph2->task_lists()->Get(j)->content()->Get(i)->sourceTaskIDs()->Get(k);
      
                EXPECT_EQ(sourceTaskIDs_graph1, sourceTaskIDs_graph2);
            }
        }
    }
}

TEST(graphFile, task_lists_TaskList_content_Task_associated_barriers_BarrierReference_wait_barriers)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
        auto content_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables
       
        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) { //For each Task Table

            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1->task_lists()->Get(j)->content()->Get(i)->associated_barriers(), BarrierReference::VT_WAIT_BARRIER)); //is wait barrier present graph1
            ASSERT_TRUE(flatbuffers::IsFieldPresent(graph2->task_lists()->Get(j)->content()->Get(i)->associated_barriers(), BarrierReference::VT_WAIT_BARRIER)); //is wait barrier present graph2

            auto wait_barrier_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->associated_barriers()->wait_barrier(); //get wait_barrier
            auto wait_barrier_graph2 = graph2->task_lists()->Get(j)->content()->Get(i)->associated_barriers()->wait_barrier(); //get wait_barrier
      
            EXPECT_EQ(wait_barrier_graph1,wait_barrier_graph2);
        }
    }
}

TEST(graphFile, task_lists_TaskList_content_Task_associated_barriers_BarrierReference_update_barriers)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    //Is task_lists present in graph1
    auto taskListPresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_TASK_LISTS);

    //if present in graph1
    if(taskListPresentGraph1) {

        for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
            auto task_tables_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables
       
                for (flatbuffers::uoffset_t i = 0; i < task_tables_size; i++) { //For each Task Table

                //check if update_barriers is present 
                auto update_barriers_present = flatbuffers::IsFieldPresent(graph1->task_lists()->Get(j)->content()->Get(i)->associated_barriers(),BarrierReference::VT_UPDATE_BARRIERS);

                if(update_barriers_present) { //If present
                
                    auto update_barriers_size = graph1->task_lists()->Get(j)->content()->Get(i)->associated_barriers()->update_barriers()->size(); //Get Size of update_barriers vector
                        
                        for (flatbuffers::uoffset_t k = 0; k < update_barriers_size ; k++) { //For each update_barriers element check if equal
            
                        auto update_barriers_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->associated_barriers()->update_barriers()->Get(k); //get update_barriers
                        auto update_barriers_graph2 = graph2->task_lists()->Get(j)->content()->Get(i)->associated_barriers()->update_barriers()->Get(k); //get update_barriers

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

TEST(graphFile, task_lists_TaskList_content_Task_task_SpecificTask)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
        auto content_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables
       
        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) { //For each Task Table

            auto task_type_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->task_type(); //Note This returns an int and not a task type name as a string
            auto task_type_graph2 = graph2->task_lists()->Get(j)->content()->Get(i)->task_type();
    
            EXPECT_EQ(task_type_graph1,task_type_graph2);
        }
    }
}

TEST(graphFile, task_lists_TaskList_content_Task_task_SpecificTask_ControllerTask_task_ControllerSubTask_BarrierConfigurationTask_barrier_id)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
        auto content_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables
       
        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) { //For each Task Table

            auto task_type_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->task_type(); //Note This returns an int and not a task type name as a string

            if(task_type_graph1 == SpecificTask_ControllerTask) { //Is it a controller task - if not proceed to SUCEED()
                
                auto controllerSubTask = static_cast<const ControllerTask*>(graph1->task_lists()->Get(j)->content()->Get(i)->task()); // Requires `static_cast`
                
                auto controllerSubTask_type = controllerSubTask->task_type();

                if(controllerSubTask_type == ControllerSubTask_BarrierConfigurationTask) {//Is it BarrierConfigurationTask - if not proceed to SUCEED()

                    auto barrierConfigurationTask_graph1 = static_cast<const BarrierConfigurationTask*>(graph1->task_lists()->Get(j)->content()->Get(i)->task_as_ControllerTask()->task_as_BarrierConfigurationTask()); // Requires `static_cast`
                    auto barrierConfigurationTask_graph2 = static_cast<const BarrierConfigurationTask*>(graph2->task_lists()->Get(j)->content()->Get(i)->task_as_ControllerTask()->task_as_BarrierConfigurationTask()); // Requires `static_cast`

                    auto barrier_id_graph1 = barrierConfigurationTask_graph1->target()->barrier_id();
                    auto barrier_id_graph2 = barrierConfigurationTask_graph2->target()->barrier_id();

                    EXPECT_EQ(barrier_id_graph1,barrier_id_graph2); //test if barrier IDs are equal                                                
                }
                else
                    SUCCEED(); //it is not a BarrierConfigurationTask  -> pass test
            }
            else
                SUCCEED(); //it is not a controller task -> pass test
        }
    }
}

TEST(graphFile, task_lists_TaskList_content_Task_task_SpecificTask_ControllerTask_task_ControllerSubTask_BarrierConfigurationTask_consumer_count)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
        auto content_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables
       
        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) { //For each Task Table

            auto task_type_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->task_type(); //Note This returns an int and not a task type name as a string

            if(task_type_graph1 == SpecificTask_ControllerTask) { //Is it a controller task - if not proceed to SUCEED()
                
                auto controllerSubTask = static_cast<const ControllerTask*>(graph1->task_lists()->Get(j)->content()->Get(i)->task()); // Requires `static_cast`
                
                auto controllerSubTask_type = controllerSubTask->task_type();

                if(controllerSubTask_type == ControllerSubTask_BarrierConfigurationTask) {//Is it BarrierConfigurationTask - if not proceed to SUCEED()

                    auto barrierConfigurationTask_graph1 = static_cast<const BarrierConfigurationTask*>(graph1->task_lists()->Get(j)->content()->Get(i)->task_as_ControllerTask()->task_as_BarrierConfigurationTask()); // Requires `static_cast`
                    auto barrierConfigurationTask_graph2 = static_cast<const BarrierConfigurationTask*>(graph2->task_lists()->Get(j)->content()->Get(i)->task_as_ControllerTask()->task_as_BarrierConfigurationTask()); // Requires `static_cast`

                    auto consumer_count_graph1 = barrierConfigurationTask_graph1->target()->consumer_count();
                    auto consumer_count_graph2 = barrierConfigurationTask_graph2->target()->consumer_count();

                    EXPECT_EQ(consumer_count_graph1,consumer_count_graph2); //test if barrier IDs are equal                                                           
                }
                else
                    SUCCEED(); //it is not a BarrierConfigurationTask  -> pass test
            }
            else
                SUCCEED(); //it is not a controller task -> pass test
        }
    }
}

TEST(graphFile, task_lists_TaskList_content_Task_task_SpecificTask_ControllerTask_task_ControllerSubTask_BarrierConfigurationTask_producer_count)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());

    for (flatbuffers::uoffset_t j = 0; j < graph1->task_lists()->size(); j++) { //No. of TaskList tables
        
        auto content_size = graph1->task_lists()->Get(j)->content()->size(); //No. of Task tables
       
        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) { //For each Task Table

            auto task_type_graph1 = graph1->task_lists()->Get(j)->content()->Get(i)->task_type(); //Note This returns an int and not a task type name as a string

            if(task_type_graph1 == SpecificTask_ControllerTask) { //Is it a controller task - if not proceed to SUCEED()
                
                auto controllerSubTask = static_cast<const ControllerTask*>(graph1->task_lists()->Get(j)->content()->Get(i)->task()); // Requires `static_cast`
                
                auto controllerSubTask_type = controllerSubTask->task_type();

                if(controllerSubTask_type == ControllerSubTask_BarrierConfigurationTask) {//Is it BarrierConfigurationTask - if not proceed to SUCEED()

                    auto barrierConfigurationTask_graph1 = static_cast<const BarrierConfigurationTask*>(graph1->task_lists()->Get(j)->content()->Get(i)->task_as_ControllerTask()->task_as_BarrierConfigurationTask()); // Requires `static_cast`
                    auto barrierConfigurationTask_graph2 = static_cast<const BarrierConfigurationTask*>(graph2->task_lists()->Get(j)->content()->Get(i)->task_as_ControllerTask()->task_as_BarrierConfigurationTask()); // Requires `static_cast`

                    auto producer_count_graph1 = barrierConfigurationTask_graph1->target()->producer_count();
                    auto producer_count_graph2 = barrierConfigurationTask_graph2->target()->producer_count();

                    EXPECT_EQ(producer_count_graph1,producer_count_graph2); //test if barrier IDs are equal                                                           
                }
                else
                    SUCCEED(); //it is not a BarrierConfigurationTask  -> pass test
            }
            else
                SUCCEED(); //it is not a controller task -> pass test
        }
    }
}


TEST(graphFile, binary_data_BinaryData_u8)		
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    bool u8PresentGraph2 = false;
    const flatbuffers::Vector<uint8_t> * binaryData_u8_graph2 = nullptr;
    flatbuffers::uoffset_t j; 

    //Is  binary_data present in graph1
    auto  binaryDataPresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_BINARY_DATA);
    
    ASSERT_TRUE(flatbuffers::IsFieldPresent(graph1, GraphFile::VT_BINARY_DATA));

    if(binaryDataPresentGraph1) { //if present

        for (j = 0; j < graph1->binary_data()->size(); j++) { //No. of Barrier tables
    
            //is u8 present
            auto u8PresentGraph1 = flatbuffers::IsFieldPresent(graph1->binary_data()->Get(j), BinaryData::VT_U8);

            if(u8PresentGraph1) {

                //check if u8 is present in graph2 (may no tbe at the same index - check all indices)
                for (flatbuffers::uoffset_t k = 0; k < graph1->binary_data()->size(); k++){

                    if(flatbuffers::IsFieldPresent(graph2->binary_data()->Get(k), BinaryData::VT_U8)){
                        std::cout << "Found u8 in graph2" << std::endl;
                        std::cout << "Check that it is same size as graph1 (could be multiple)" << std::endl;
                        if(graph2->binary_data()->Get(k)->u8()->size() == graph1->binary_data()->Get(j)->u8()->size()) {
                            u8PresentGraph2 = true; //found u8
                            binaryData_u8_graph2 = graph2->binary_data()->Get(k)->u8();
                            break;
                        }
                    }
                }
                    
                ASSERT_TRUE(u8PresentGraph2);
            
                auto binaryData_u8_graph1 = graph1->binary_data()->Get(j)->u8();

                for (auto it = binaryData_u8_graph1->begin(); it != binaryData_u8_graph1->end(); ++it) {
                    auto indx = it - binaryData_u8_graph1->begin();
                    EXPECT_EQ(*it, binaryData_u8_graph2->Get(indx));  // Use bounds-check.
                }
            }
            else 
                SUCCEED();
        }
    }
    else 
        SUCCEED();
}

TEST(graphFile, binary_data_BinaryData_fp16)		
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    bool fp16PresentGraph2 = false;
    const flatbuffers::Vector<int16_t> * binaryData_fp16_graph2 = nullptr;
    flatbuffers::uoffset_t j; 

    //Is  binary_data present in graph1
    auto  binaryDataPresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_BINARY_DATA);


    if(binaryDataPresentGraph1) { //if present

        for (j = 0; j < graph1->binary_data()->size(); j++) { //No. of Barrier tables
    
            //is fp16 present
            auto fp16PresentGraph1 = flatbuffers::IsFieldPresent(graph1->binary_data()->Get(j), BinaryData::VT_FP16);

            if(fp16PresentGraph1) {

                //check if fp16 is present in graph2 (may no tbe at the same index - check all indices)
                for (flatbuffers::uoffset_t k = 0; k < graph1->binary_data()->size(); k++){

                    if(flatbuffers::IsFieldPresent(graph2->binary_data()->Get(k), BinaryData::VT_FP16)){
                        std::cout << "Found fp16 in graph2" << std::endl;
                        std::cout << "Check that it is same size as graph1 (could be multiple)" << std::endl;
                        if(graph2->binary_data()->Get(k)->fp16()->size() == graph1->binary_data()->Get(j)->fp16()->size()) {
                            fp16PresentGraph2 = true; //found u8
                            binaryData_fp16_graph2 = graph2->binary_data()->Get(k)->fp16();
                            break;
                        }
                    }
                }
                    
                ASSERT_TRUE(fp16PresentGraph2);
            
                auto binaryData_fp16_graph1 = graph1->binary_data()->Get(j)->fp16();

                for (auto it = binaryData_fp16_graph1->begin(); it != binaryData_fp16_graph1->end(); ++it) {
                    auto indx = it - binaryData_fp16_graph1->begin();
                    EXPECT_EQ(*it, binaryData_fp16_graph2->Get(indx));  // Use bounds-check.
                }
            }
            else 
                SUCCEED();
        }
    }
    else 
        SUCCEED();
}

TEST(graphFile, binary_data_BinaryData_i32)		
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    bool i32PresentGraph2 = false;
    const flatbuffers::Vector<int32_t> * binaryData_i32_graph2 = nullptr;
    flatbuffers::uoffset_t j; 

    //Is  binary_data present in graph1
    auto  binaryDataPresentGraph1 = flatbuffers::IsFieldPresent(graph1, GraphFile::VT_BINARY_DATA);

    if(binaryDataPresentGraph1) { //if present

        for (j = 0; j < graph1->binary_data()->size(); j++) { //No. of Barrier tables
    
            //is i32 present
            auto i32PresentGraph1 = flatbuffers::IsFieldPresent(graph1->binary_data()->Get(j), BinaryData::VT_I32);

            if(i32PresentGraph1) {

                //check if i32 is present in graph2 (may no tbe at the same index - check all indices)
                for (flatbuffers::uoffset_t k = 0; k < graph1->binary_data()->size(); k++){

                    if(flatbuffers::IsFieldPresent(graph2->binary_data()->Get(k), BinaryData::VT_I32)){
                        std::cout << "Found i32 in graph2" << std::endl;
                        std::cout << "Check that it is same size as graph1 (could be multiple)" << std::endl;
                        if(graph2->binary_data()->Get(k)->i32()->size() == graph1->binary_data()->Get(j)->i32()->size()) {
                            i32PresentGraph2 = true; //found u8
                            binaryData_i32_graph2 = graph2->binary_data()->Get(k)->i32();
                            break;
                        }
                    }
                }
                    
                ASSERT_TRUE(i32PresentGraph2);
            
                auto binaryData_i32_graph1 = graph1->binary_data()->Get(j)->i32();

                for (auto it = binaryData_i32_graph1->begin(); it != binaryData_i32_graph1->end(); ++it) {
                    auto indx = it - binaryData_i32_graph1->begin();
                    EXPECT_EQ(*it, binaryData_i32_graph2->Get(indx));  // Use bounds-check.
                }
            }
            else 
                SUCCEED();
        }
    }
    else 
        SUCCEED();
}

TEST(graphFile, header_SummaryHeader_task_count)	
{
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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
    Blob blob_1(file_name_one.c_str());
    Blob blob_2(file_name_two.c_str());
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

int main(int argc, char **argv) {

  std::string command_line_arg_one(argc == 3 ? argv[1] : "");
  std::string command_line_arg_two(argc == 3 ? argv[2] : "");
  testing::InitGoogleTest(&argc, argv);
  //testing::AddGlobalTestEnvironment(new TestEnvironment(command_line_arg_one, command_line_arg_two));
  testing::AddGlobalTestEnvironment(new TestEnvironment("/home/john/blobv3/try4.blob", "/home/john/blobv3/try5.blob"));
  return RUN_ALL_TESTS();
}
