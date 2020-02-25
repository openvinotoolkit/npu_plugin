#include "gtest/gtest.h"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "tools/graph_comparator/include/graph_comparator/graph_comparator.hpp"

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

TEST(runtime_model, test_soh_dma_addresses)
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({416,416,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1#34");

    auto pool0 = om.maxPool(input0, {2, 2}, {2, 2}, {0, 0, 0, 0}, true, "", "floor", mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool1/max_pool#35");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*16*32);
    auto weights0 = om.constantInt(weightsData0,{3,3,16,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.002742463955655694},{-0.32530343532562256},{0.374024897813797}}, "conv2#4_weights#5");
    auto conv0 = om.conv(pool0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv2#36");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.075476120604435e-05},{-inf},{inf}}, "conv2#4_bias#6");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool1 = om.maxPool(bias_c0, {2, 2}, {2, 2}, {0, 0, 0, 0}, true, "", "floor", mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool2/max_pool#37");

    om.output(pool1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-PrefetchAdaptive.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    mv::DataModel dm(om);
    mv::tools::GraphComparator gc;
    std::string outputBlobFile = "./output/mcm.blob"; // generated blob file

    //XXX: buffer needs to be deleted after the test, or else there will be a leak
    char* buffer = 0;
    const MVCNN::GraphFileT& blob = gc.loadGraphFile(outputBlobFile, buffer);

    std::vector<mv::Data::OpListIterator> dpuTasks;
    for (auto op = om.opBegin(); op != om.opEnd(); ++op)
    {
        if (op->getOpType() == "DPUTask" && op->get<std::string>("splitStrategy") == "SplitOverH")
        {
            dpuTasks.push_back(op);
        }
    }

    for (auto dpuTask: dpuTasks)
    {
        auto inputTensor = dpuTask->getInputTensor(0);
        auto dmaTask = om.getSourceOp(inputTensor);
        auto dmaTaskSrcTensor = dmaTask->getInputTensor(0);
        auto dmaTaskDstTensor = inputTensor;

        for (std::size_t i = 0; i < blob.task_lists.size(); ++i)
        {
            const std::unique_ptr<MVCNN::TaskListT>& task_list = blob.task_lists[i];
            for (std::size_t j = 0; j < task_list->content.size(); j++)
            {
                const std::unique_ptr<MVCNN::TaskT>& task = task_list->content[j];
                if (task->name == dmaTask->getName() && task->task.type == MVCNN::SpecificTask::SpecificTask_NNDMATask)
                {
                    const MVCNN::NNDMATaskT& blobDmaTask = *task->task.AsNNDMATask();
                    for (int k = 0; k < 2; k++)
                    {
                        std::string blobTaskName;
                        mv::Data::TensorIterator expectedTensor;
                        MVCNN::IndirectDataReferenceT *blobTensorIndirectRef;
                        if (k == 0)
                        {
                            blobTaskName = blobDmaTask.src->name;
                            expectedTensor = dmaTaskSrcTensor;
                            blobTensorIndirectRef = blobDmaTask.src->data.get();
                        }
                        else
                        {
                            blobTaskName = blobDmaTask.dst->name;
                            expectedTensor = inputTensor;
                            blobTensorIndirectRef = blobDmaTask.dst->data.get();
                        }

                        std::string subTensorId = blobTaskName.substr(blobTaskName.find("sub") + 3);
                        std::size_t clusterId = std::stoi(subTensorId);

                        auto subtensor = expectedTensor->getSubTensor(clusterId);
                        auto tensorAllocators = expectedTensor->get<std::set<std::string>>("allocators");
                        auto tensorAllocatorName = tensorAllocators.begin();
                        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);

                        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, expectedTensor);
                        std::vector<uint32_t> dimensions = subtensor.getShape();

                        if (*tensorAllocatorName == "GraphFile")
                        {
                            if(!dmaTaskSrcTensor->isSparse())
                            {

                                auto offset = subtensor.get<std::vector<std::size_t>>("offset");
                                auto index = expectedTensor->getOrder().subToInd(expectedTensor->getShape(), offset);
                                auto byte_index = index * expectedTensor->getDType().getSizeInBits() / 8;

                                ASSERT_EQ(blobTensorIndirectRef->data_index, byte_index);
                            }
                            else
                            {
                                ASSERT_EQ(blobTensorIndirectRef->data_index, 0);
                            }
                        }
                        else if(*tensorAllocatorName == "ProgrammableInput" || *tensorAllocatorName == "ProgrammableOutput" ||
                                *tensorAllocatorName == "VPU_DDR_BSS" || *tensorAllocatorName == "VPU_DDR_Heap")
                        {
                            auto offset = subtensor.get<std::vector<std::size_t>>("offset");
                            auto index = expectedTensor->getOrder().subToInd(expectedTensor->getShape(), offset);
                            auto byte_index = index * expectedTensor->getDType().getSizeInBits() / 8;

                            auto starting_address = 0;
                            if(expectedTensor->hasAttr("address"))
                                starting_address = expectedTensor->get<std::size_t>("address");
                            else
                            {
                                auto masterBuffer = tensorAllocator.getTopMasterBuffer(tensorBufferIt);
                                starting_address = (*masterBuffer)->getOffset();
                            }

                            ASSERT_EQ(blobTensorIndirectRef->data_index, starting_address + byte_index);

                        }
                        else
                        {
                            if(expectedTensor->hasAttr("address"))
                                ASSERT_EQ(blobTensorIndirectRef->data_index, subtensor.getAddress());
                            else
                                ASSERT_EQ(blobTensorIndirectRef->data_index, tensorBufferIt->getOffset());
                        }
                    }

                }
            }
        }
    }

    if (buffer)
        delete [] buffer;

}
