#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

void run_test(const std::string& dataType, bool useQuantization)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    mv::Data::TensorIterator input;
    if (!useQuantization)
    {
        input = om.input({16, 16, 1, 1}, mv::DType(dataType), mv::Order("NCHW"));
    }
    else
    {
        std::vector<int64_t> zp = { 0 };
        std::vector<double> scale(1, 0.42 );
        std::vector<double> min = { 1 };
        std::vector<double> max = { 1 };

        mv::QuantizationParams inputQuantParams(zp, scale, min, max);
        input = om.input({16, 16, 1, 1}, mv::DType(dataType), mv::Order("NCHW"), inputQuantParams);
    }

    short unsigned int kH = 2;
    short unsigned int kW = 2;
    auto pool = om.averagePool(input, {kH, kW}, {2, 2}, {1, 1, 1, 1});
    auto poolShape = pool->getShape();
    auto output = om.output(pool);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);

    auto& compDesc = unit.compilationDescriptor();
    compDesc.clear();
    compDesc.addGroup("root");
    compDesc.addToGroup("root", "AverageAsDepthWise", "Singular", false);

    unit.initialize();
    unit.run();

    auto depthWiseOp = om.getOps("DepthwiseConv");

    ASSERT_EQ(depthWiseOp.size(), 1);
    auto op = depthWiseOp[0];

    if (useQuantization)
    {
        ASSERT_TRUE(op->hasAttr("quantParams"));
        auto weightsTensor = op->getInputTensor(1);
        auto quant = weightsTensor->get<mv::QuantizationParams>("quantParams");
        auto scale = quant.get<std::vector<double>>("scale");
        ASSERT_EQ(scale.size(), 1);
        ASSERT_EQ(scale[0], 1/((double)kH * (double)kW));
    }
    else
    {
        auto weights = op->getInputTensor(1);
        auto expectedShape = mv::Shape({kH, kW, input->getShape()[mv::IO_CHANNEL_DIMENSION], 1});
        ASSERT_TRUE(weights->getShape() == expectedShape);
        ASSERT_EQ(weights->getData()[0], 1/((double)kH * (double)kW));
    }
}

TEST(ReplacementPasses, AveragePoolAsDepthwise)
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);

    bool useQuantization = true;
    run_test("Int8", useQuantization);
    run_test("Float16", !useQuantization);

    // XXX: Do we need to run this test also: input is float, but weights are quantized?
    // run_test("Float16", useQuantization);
    // std::cout << "--------------" << std::endl;

    mv::Logger::setVerboseLevel(mv::VerboseLevel::Warning);
}
