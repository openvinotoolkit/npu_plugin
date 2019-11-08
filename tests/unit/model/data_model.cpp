#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(data_model, allocate_unpopulated_tensor)
{

    /*mv::OpModel om("TestModel");
    mv::ControlModel cm(om);
    mv::DataModel dm(om);


    auto input = om.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto pool1 = om.maxPool(input, {3, 3}, {1, 1}, {1, 1, 1, 1});
    auto pool1Op = om.getSourceOp(pool1);
    auto pool2 = om.maxPool(pool1, {3, 3}, {1, 1}, {1, 1, 1, 1});
    auto pool2Op = om.getSourceOp(pool2);
    om.output(pool2);

    auto stage = cm.addStage();
    cm.addToStage(stage, pool1Op);
    cm.addToStage(stage, pool2Op);
    dm.addAllocator("Memory1", 4096, 0, 2);
    auto buf1 = dm.allocateTensor("Memory1", stage, pool1);
    auto buf2 = dm.allocateTensor("Memory1", stage, pool2);
    std::cout << buf1->toString() << std::endl;
    std::cout << buf2->toString() << std::endl;

    std::cout << pool1->toString() << std::endl;

    for (auto bufIt = dm.bufferBegin("Memory1", stage); bufIt != dm.bufferEnd("Memory1", stage); ++bufIt)
    {
        std::cout << bufIt->toString() << std::endl;
    }

    std::cout << dm.getBuffer("Memory1", stage, pool1)->toString() << std::endl;*/

}

TEST(data_model, allocate_populated_tensor)
{

    /*mv::OpModel om("TestModel");
    mv::ControlModel cm(om);
    mv::DataModel dm(om);

    auto input = om.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto weightsData = mv::utils::generateSequence<double>(3 * 3 * 3 * 3, 1.0f, 0.01f);
    auto weights = om.constant(weightsData, {3, 3, 3, 3}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)));
    auto conv1 = om.conv(input, weights, {1, 1}, {1, 1, 1, 1});
    auto conv1Op = om.getSourceOp(conv1);
    om.output(conv1);

    auto stage = cm.addStage();
    cm.addToStage(stage, conv1Op);
    dm.addAllocator("Memory1", 4096, 0, 2);
    auto buf = dm.allocateTensor("Memory1", stage, weights);

    std::cout << buf->toString(true) << std::endl;*/

}
