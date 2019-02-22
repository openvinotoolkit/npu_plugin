#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(sparsity, case_cm)
{
    //This test covers the ChannelMajor Convolution case
    //mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Debug);
    mv::OpModel om("testModel");

    auto input = om.input({56, 56 , 3}, mv::DType("UInt8"), mv::Order("CWH"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*64);
    auto weights = om.constant(weightsData, {3, 3, 3, 64}, mv::DType("UInt8"), mv::Order(mv::Order::getColMajorID(4)), "weights");
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1); //stride {1,1}
    auto convOp = om.getSourceOp(conv);

    mv::Element dummyCompDesc("dummyPassDesc");

    mv::json::Object compOutput;
    mv::TargetDescriptor desc;

    desc.setTarget(mv::Target::ma2490);

    mv::pass::PassRegistry::instance().find("MarkHardwareOperations")->run(om, desc, dummyCompDesc, compOutput);
    mv::pass::PassRegistry::instance().find("Sparsity")->run(om, desc, dummyCompDesc, compOutput);

    //ref data is based on result on POC test res2a_branch2a/quantized_model.tflite
    mv::DataModel dm(om);
    auto resData = dm.getTensor(convOp->get<std::string>("sparsityMap"))->getData();

    std::ifstream outputfile(mv::utils::projectRootPath() + std::string("/tests/data/res1_sparsity_map.bin"), std::ios::binary );

    unsigned count = 0;
    uint8_t a;
    while(outputfile.read(reinterpret_cast<char *>(&a), sizeof(a)))
    {
        if ( a != resData[count])
        {
            std::cout << count << std::endl;
            std::cout << "ref " << (int) a << "  res " << std::to_string(resData[count]) << std::endl;
        }
        ASSERT_TRUE(a == resData[count]);
        count++;
    }

    ASSERT_TRUE(count == resData.size());
}

TEST(sparsity, case_hwPooling)
{
    //This test covers the depthWiseConv Convolution case
    //mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Debug);
    mv::OpModel om("testModel");

    auto input = om.input({56, 56 , 2048}, mv::DType("UInt8"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*1*2048);
    auto weights = om.constant(weightsData, {1, 1, 2048, 1}, mv::DType("UInt8"), mv::Order(mv::Order::getColMajorID(4)), "weights");
    //auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1); //stride {1,1}

    auto pool = om.maxPool(input, {1,1}, {1, 1}, {0, 0, 0, 0}); //stride {1,1}
    auto poolOp = om.getSourceOp(pool);
    auto output = om.output(pool);
    mv::Element dummyPassDesc("dummyPassDesc");
    mv::json::Object compOutput;
    mv::TargetDescriptor desc;

    desc.setTarget(mv::Target::ma2490);

    mv::pass::PassRegistry::instance().find("MarkHardwareOperations")->run(om, desc, dummyPassDesc, compOutput);
    mv::pass::PassRegistry::instance().find("Sparsity")->run(om, desc, dummyPassDesc, compOutput);

    //ref data is based on result on POC test res5c/quantized_model.tflite
    mv::DataModel dm(om);
    auto resData = dm.getTensor(poolOp->get<std::string>("sparsityMap"))->getData();

    std::ifstream outputfile(mv::utils::projectRootPath() + std::string("/tests/data/res5c_sparsity_map.bin"), std::ios::binary );

    unsigned count = 0;
    uint8_t a;
    while(outputfile.read(reinterpret_cast<char *>(&a), sizeof(a)))
    {
        if ( a != resData[count])
        {
            std::cout << count << std::endl;
            std::cout << "ref " << (int) a << "  res " << std::to_string((int)resData[count]) << std::endl;
        }
        ASSERT_TRUE(a == resData[count]);
        count++;
    }

    ASSERT_TRUE(count == resData.size());
}