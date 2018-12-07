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

TEST(graphFile, summaryHeader_Version)	
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


TEST(graphFile, summaryHeader_task_count)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    
    EXPECT_EQ(graph1->header()->task_count(),graph2->header()->task_count());
}

TEST(graphFile, summaryHeader_layer_count)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    
    EXPECT_EQ(graph1->header()->layer_count(),graph2->header()->layer_count());
}


TEST(graphFile, summaryHeader_net_input)	
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

TEST(graphFile, summaryHeader_net_output)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    

    auto net_output_graph1 = graph1->header()->net_output();
    auto net_output_graph2 = graph2->header()->net_output();

    
    EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));
    EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));
    EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));
    EXPECT_EQ(net_output_graph1->Get(0)->dimensions()->Get(0),net_output_graph2->Get(0)->dimensions()->Get(0));

}

TEST(graphFile, summaryHeader_resources)	
{
    Blob blob_1("/home/john/blobv3/vpu_3_1.bin");
    Blob blob_2("/home/john/blobv3/vpu_3_1.bin");
    const auto graph1 = GetGraphFile(blob_1.get_ptr());
    const auto graph2 = GetGraphFile(blob_2.get_ptr());
    

    auto resources_graph1 = graph1->header()->resources();
    auto resources_graph2 = graph2->header()->resources();

    
    EXPECT_EQ(resources_graph1->shave_mask(),resources_graph1->shave_mask());
    EXPECT_EQ(resources_graph1->nce1_mask(),resources_graph1->nce1_mask());
    EXPECT_EQ(resources_graph1->dpu_mask(),resources_graph1->dpu_mask());
    EXPECT_EQ(resources_graph1->leon_cmx(),resources_graph1->leon_cmx());
    EXPECT_EQ(resources_graph1->nn_cmx(),resources_graph1->nn_cmx());
    EXPECT_EQ(resources_graph1->ddr_scratch(),resources_graph1->ddr_scratch());

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}