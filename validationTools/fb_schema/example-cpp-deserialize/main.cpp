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

TEST(graphFile, header_SummaryHeader_version_Version)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    
    EXPECT_EQ(graph1->header()->version()->majorV(),graph2->header()->version()->majorV());
    EXPECT_EQ(graph1->header()->version()->minorV(),graph2->header()->version()->minorV());
    EXPECT_EQ(graph1->header()->version()->patchV(),graph2->header()->version()->patchV());
    EXPECT_EQ(graph1->header()->version()->hash()->c_str(),graph2->header()->version()->hash()->c_str());
}


TEST(DISABLE_graphFile, header_SummaryHeader_task_count)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    
    EXPECT_EQ(graph1->header()->task_count(),graph2->header()->task_count());
}

TEST(DISABLE_graphFile, header_SummaryHeader_layer_count)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    
    EXPECT_EQ(graph1->header()->layer_count(),graph2->header()->layer_count());
}

//Need to fix
TEST(DISABLE_graphFile, header_SummaryHeader_net_input_TensorReference_dimensions)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    

    auto net_input_graph1 = graph1->header()->net_input();
    auto net_input_graph2 = graph2->header()->net_input();

    
    EXPECT_EQ(net_input_graph1->Get(0)->dimensions()->Get(0),net_input_graph2->Get(0)->dimensions()->Get(0));
    EXPECT_EQ(net_input_graph1->Get(0)->dimensions()->Get(0),net_input_graph2->Get(0)->dimensions()->Get(0));
    EXPECT_EQ(net_input_graph1->Get(0)->dimensions()->Get(0),net_input_graph2->Get(0)->dimensions()->Get(0));
    EXPECT_EQ(net_input_graph1->Get(0)->dimensions()->Get(0),net_input_graph2->Get(0)->dimensions()->Get(0));

}

// TEST(DISABLE_graphFile, summaryHeader_net_output)	
// {
//     Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
//     Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
//     const auto graph1 = GetGraphFile(blob_1.get_ptr());
//     const auto graph2 = GetGraphFile(blob_2.get_ptr());
    

//     auto net_output_graph1 = graph1->header()->net_output();
//     auto net_output_graph2 = graph2->header()->net_output();

    
//     EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));
//     EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));
//     EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));
//     EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));

// }

// TEST(DISABLE_graphFile, summaryHeader_resources)	
// {
//     Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
//     Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
//     const auto graph1 = GetGraphFile(blob_1.get_ptr());
//     const auto graph2 = GetGraphFile(blob_2.get_ptr());
    

//     auto resources_graph1 = graph1->header()->resources();
//     auto resources_graph2 = graph2->header()->resources();

    
    
//     EXPECT_EQ(resources_graph1->nce1_mask(),resources_graph1->nce1_mask());
//     EXPECT_EQ(resources_graph1->dpu_mask(),resources_graph1->dpu_mask());
//     EXPECT_EQ(resources_graph1->leon_cmx(),resources_graph1->leon_cmx());
//     EXPECT_EQ(resources_graph1->nn_cmx(),resources_graph1->nn_cmx());
//     EXPECT_EQ(resources_graph1->ddr_scratch(),resources_graph1->ddr_scratch());
// }

// TEST(DISABLE_graphFile, summaryHeader_original_strucure)	
// {
//     Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
//     Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
//     const auto graph1 = GetGraphFile(blob_1.get_ptr());
//     const auto graph2 = GetGraphFile(blob_2.get_ptr());

//     auto original_structure_graph1 = graph1->header()->original_structure();
//     auto original_structure_graph2 = graph2->header()->original_structure();

// }

// TEST(DISABLE_graphFile, task_lists_nodeID)	// TEST(DISABLE_graphFile, task_lists_sourceTaskIDs)	
// {
//     Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
//     Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
//     const auto graph1 = GetGraphFile(blob_1.get_ptr());
//     const auto graph2 = GetGraphFile(blob_2.get_ptr());

//     auto task_lists_graph1 = graph1->task_lists();
//     auto task_lists_graph2 = graph2->task_lists();


//     std::cout << "task_lists_size  " << task_lists_graph1->size() << std::endl; 


//      for (flatbuffers::uoffset_t i = 0; i < task_lists_graph1->size(); i++) {
//       auto sourceTaskIDs_graph1 = task_lists_graph1->Get(i)->content()->Get(i)->sourceTaskIDs()->Get(i);
//       auto sourceTaskIDs_graph2 = task_lists_graph2->Get(i)->content()->Get(i)->sourceTaskIDs()->Get(i);
//       EXPECT_EQ(sourceTaskIDs_graph1,sourceTaskIDs_graph2);
//     }

// }
// {
//     Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
//     Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
//     const auto graph1 = GetGraphFile(blob_1.get_ptr());
//     const auto graph2 = GetGraphFile(blob_2.get_ptr());

//     auto task_lists_graph1 = graph1->task_lists();
//     auto task_lists_graph2 = graph2->task_lists();


//     std::cout << "task_lists_size  " << task_lists_graph1->size() << std::endl; 

//     for (flatbuffers::uoffset_t i = 0; i < task_lists_graph1->size() - 1; i++) {
//       auto nodeID_graph1 = task_lists_graph1->Get(i)->content()->Get(i)->nodeID();
//       auto nodeID_graph2 = task_lists_graph2->Get(i)->content()->Get(i)->nodeID();
//       EXPECT_EQ(nodeID_graph1,nodeID_graph1);
//     }


// }

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
      
            EXPECT_EQ(sourceTaskIDs_graph1, sourceTaskIDs_graph1);
            }
        }
    }
}


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
       
        for (flatbuffers::uoffset_t i = 0; i < content_size; i++) {
            auto wait_barrier_graph1 = TaskLists_graph1->Get(j)->content()->Get(i)->associated_barriers()->wait_barrier(); //get wait_barrier
            auto wait_barrier_graph2 = TaskLists_graph2->Get(j)->content()->Get(i)->associated_barriers()->wait_barrier(); //get wait_barrier
      
            std::cout << "wait_barrier_graph1 " << wait_barrier_graph1 << std::endl;
            std::cout << "wait_barrier_graph2 " << wait_barrier_graph2 << std::endl;

            EXPECT_EQ(wait_barrier_graph1,wait_barrier_graph2);
        }
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}