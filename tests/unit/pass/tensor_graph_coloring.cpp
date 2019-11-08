#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/utils/custom_strings.hpp"

TEST(graph_coloring, single_conv)
{
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({112, 224, 3, 1}, mv::DType("UInt8"), mv::Order("NCHW"));
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(7*7*3*64);
    auto weights = om.constantInt(weightsData, {7, 7, 3, 64}, mv::DType("UInt8"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {2, 2}, {3, 3, 3, 3});

    om.output(conv);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";

    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateSparsityMaps", "enableRealSparsity", true);

    compDesc.remove("finalize","GenerateWorkloads");
    compDesc.remove("serialize","GenerateBlobKmb");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    std::map<std::string, std::pair<uint64_t, int64_t>> refTensorSizeAddress;
    //since neighbor weights has changed to include node weight, in this simple network
    // where everything is connected, the order will not matter (all have the same neighbors weight)
    // so the addresses here are not matching with POC.
    //CMX
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createAlignConstantName("ConstantInt_0"))+":0"] = std::make_pair(10240, 557056);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createFakeSparsityMapName(mv::createDPUTaskName("Conv_0")))+":0"] = std::make_pair(4096, 552960);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createWeightTableName(mv::createDPUTaskName("Conv_0")))+":0"] = std::make_pair(1024, 551936);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName("Input_0")+":0"] = std::make_pair(75264, 476672);
    refTensorSizeAddress[mv::createDPUTaskName("Conv_0")+":0"] = std::make_pair(476672, 0);

    //BSS
    refTensorSizeAddress[mv::createAlignConstantName("ConstantInt_0")+":0"] = std::make_pair(10240,  5120);
    refTensorSizeAddress[mv::createFakeSparsityMapName(mv::createDPUTaskName("Conv_0"))+":0"] = std::make_pair(4096, 1024);
    refTensorSizeAddress[mv::createWeightTableName(mv::createDPUTaskName("Conv_0"))+":0"] = std::make_pair(1024, 0);

    //TODO HEAP, not yet fully implemented on POC
    refTensorSizeAddress[mv::createDMATaskNNCMX2DDRName(mv::createDPUTaskName("Conv_0"))+":0"] = std::make_pair(476672, 0);
    refTensorSizeAddress["Input_0:0"] = std::make_pair(75264, 476672);

    mv::DataModel dm(om);
    for (auto itr = refTensorSizeAddress.begin(); itr != refTensorSizeAddress.end(); itr++)
    {
        auto tensor = om.getTensor(itr->first);

        //std::cout << "checking " << itr->first << std::endl;
        auto tensorAllocatorName = tensor->get<std::set<std::string>>("allocators").begin();
        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, tensor); // 0 is the only stage for now, but this will probably change in the future
        ASSERT_EQ(tensorBufferIt->getOffset(), itr->second.second);
        ASSERT_EQ(tensor->computeTotalSize(), itr->second.first);
    }
    //system("dot -Tpng original_model.dot -o original_model.png");
    //system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
}


TEST(graph_coloring, three_conv)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({56, 56, 16, 1}, mv::DType("UInt8"), mv::Order("NHWC"));

    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*16*64);
    auto weights = om.constantInt(weightsData, {1, 1, 16, 64}, mv::DType("UInt8"), mv::Order("NCHW"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<int64_t> biasesData =  mv::utils::generateSequence<int64_t>(conv->getShape()[mv::IO_CHANNEL_DIMENSION]);
    auto biases = om.constantInt(biasesData, {conv->getShape()[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Int32"), mv::Order("W"),{{},{},{},{}}, "biases");
    auto bias = om.bias(conv, biases);

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t>(3*3*64*64);
    auto weights1 = om.constantInt(weightsData1, {3, 3, 64, 64}, mv::DType("UInt8"), mv::Order("NCHW"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {1, 1, 1, 1});

    std::vector<int64_t> biasesData1 =  mv::utils::generateSequence<int64_t>(conv1->getShape()[mv::IO_CHANNEL_DIMENSION]);
    auto biases1 = om.constantInt(biasesData1, {conv->getShape()[mv::IO_CHANNEL_DIMENSION]},mv::DType("Int32"), mv::Order("W"),{{},{},{},{}}, "biases1");
    auto bias1 = om.bias(conv1, biases1);

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t>(64*64*1*1);
    auto weights2 = om.constantInt(weightsData2, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order("NCWH"));
    auto conv2 = om.conv(conv1, weights2, {1, 1}, {0, 0, 0, 0});
    std::vector<int64_t> biasesData2 =  mv::utils::generateSequence<int64_t>(conv2->getShape()[mv::IO_CHANNEL_DIMENSION]);
    auto biases2 = om.constantInt(biasesData2, {conv->getShape()[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Int32"), mv::Order("W"),{{},{},{},{}}, "biases2");
    auto bias2 = om.bias(conv2, biases2);

    om.output(conv2);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateSparsityMaps", "enableRealSparsity", true);
    compDesc.remove("finalize", "GenerateWorkloads");
    compDesc.remove("serialize", "GenerateBlobKmb");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    std::map<std::string, std::pair<uint64_t, int64_t>> refTensorSizeAddress;
    //since neighbor weights has changed to include node weight, in this simple network
    // where everything is connected, the order will not matter (all have the same neighbors weight)
    // so the addresses here are not matching with POC.
    //CMX
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName("ConstantInt_0")+":0"] = std::make_pair(1024,77184);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createSparsityMapName("ConstantInt_0:0"))+":0"] = std::make_pair(1024, 76160);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createWeightTableName(mv::createDPUTaskName("Conv_0")))+":0"] = std::make_pair(1024,75136);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName("Input_0")+":0"] = std::make_pair(68992,6144);
    refTensorSizeAddress[mv::createDPUTaskName("Conv_0")+":0"] = std::make_pair(238336,287488);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName("ConstantInt_2")+":0"] = std::make_pair(36864,250624);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createSparsityMapName("ConstantInt_2:0"))+":0"] = std::make_pair(5120,245504);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createWeightTableName(mv::createDPUTaskName("Conv_1")))+":0"] = std::make_pair(1024,244480);
    refTensorSizeAddress[mv::createDPUTaskName("Conv_1")+":0"] = std::make_pair(238336,6144);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName("ConstantInt_4")+":0"] = std::make_pair(4096,2048);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createSparsityMapName("ConstantInt_4:0"))+":0"] = std::make_pair(1024,1024);
    refTensorSizeAddress[mv::createDMATaskDDR2NNCMXName(mv::createWeightTableName(mv::createDPUTaskName("Conv_2")))+":0"] = std::make_pair(1024,0);
    refTensorSizeAddress[mv::createDPUTaskName("Conv_2")+":0"] = std::make_pair(238336,244480);

    //BSS
    refTensorSizeAddress["ConstantInt_0:0"] = std::make_pair(1024, 51200);
    refTensorSizeAddress[mv::createSparsityMapName("ConstantInt_0:0")+":0"] = std::make_pair(1024, 50176);
    refTensorSizeAddress["ConstantInt_2:0"] = std::make_pair(36864, 13312);
    refTensorSizeAddress[mv::createSparsityMapName("ConstantInt_2:0")+":0"] = std::make_pair(5120, 8192);
    refTensorSizeAddress["ConstantInt_4:0"] = std::make_pair(4096, 4096);
    refTensorSizeAddress[mv::createSparsityMapName("ConstantInt_4:0")+":0"] = std::make_pair(1024, 3072);
    refTensorSizeAddress[mv::createWeightTableName(mv::createDPUTaskName("Conv_0"))+":0"] = std::make_pair(1024, 2048);
    refTensorSizeAddress[mv::createWeightTableName(mv::createDPUTaskName("Conv_1"))+":0"] = std::make_pair(1024, 1024);
    refTensorSizeAddress[mv::createWeightTableName(mv::createDPUTaskName("Conv_2"))+":0"] = std::make_pair(1024, 0);

    //TODO HEAP, not yet fully implemented on POC

    mv::DataModel dm(om);
    for (auto itr = refTensorSizeAddress.begin(); itr != refTensorSizeAddress.end(); itr++)
    {
        auto tensor = om.getTensor(itr->first);
//        ASSERT_TRUE(tensor->hasAttr("address"));
        auto tensorAllocatorName = tensor->get<std::set<std::string>>("allocators").begin();
        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, tensor); // 0 is the only stage for now, but this will probably change in the future
        ASSERT_EQ(tensorBufferIt->getOffset(), itr->second.second);
        ASSERT_EQ(tensor->computeTotalSize(), itr->second.first);
    }
    //system("dot -Tpng original_model.dot -o original_model.png");
    //system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
}
