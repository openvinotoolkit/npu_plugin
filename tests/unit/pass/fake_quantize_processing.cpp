#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

// Seed is given for reproducibility of results
static std::vector<double> generateRandomSequence(std::size_t dataSize, double start, double end, unsigned seed)
{
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {seed};  // Generates random integers
    std::uniform_real_distribution<double> dist {start, end};

    auto gen = [&dist, &mersenne_engine](){
        return dist(mersenne_engine);
    };

    std::vector<double> result(dataSize);
    std::generate(result.begin(), result.end(), gen);

    return result;

}

TEST(fake_quantize_proc, two_branches_one_without_fq_ssd_case)
{
    mv::OpModel om("testModel");

    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");

    auto weightsData0 = generateRandomSequence(3 * 3 * 16, -0.3, 0.3, 42);
    auto weights0 = om.constant(weightsData0,{3,3,16,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4), {{0},{1.0},{},{}}, "Weights");

    std::vector<double> minData = mv::utils::generateSequence<double> (1, *std::min_element(weightsData0.begin(), weightsData0.end()), 0);
    std::vector<double> maxData = mv::utils::generateSequence<double> (1, *std::max_element(weightsData0.begin(), weightsData0.end()), 0);

    auto weights_min0 = om.constant(minData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto weights_max0 = om.constant(maxData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto weights_min1 = om.constant(minData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto weights_max1 = om.constant(maxData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));

    auto weights_fake_quant = om.fakeQuantize(weights0, weights_min0, weights_max0, weights_min1, weights_max1, 256);

    auto conv0 = om.conv(input0, weights_fake_quant, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("Default"), {{0},{1.0},{},{}}, "conv");

    auto conv0_fq_min0 = weights_min0;
    auto conv0_fq_max0 = weights_max0;
    auto conv0_fq_min1 = weights_min1;
    auto conv0_fq_max1 = weights_max1;

    auto conv0_fake_quant = om.fakeQuantize(conv0, conv0_fq_min0, conv0_fq_max0, conv0_fq_min1, conv0_fq_max1, 255);

    auto pool0 = om.maxPool(conv0, {1, 1}, {1, 1}, {0, 0, 0, 0}, false, mv::DType("Default"), {{0},{1.0},{},{}}, "PoolLayer");

    om.output(conv0_fake_quant, mv::DType("Float16"));

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    auto fq_ops = om.getOps("FakeQuantize");
    ASSERT_FALSE(fq_ops.empty());
    mv::pass::PassRegistry::instance().find("FakeQuantize")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    fq_ops = om.getOps("FakeQuantize");
    ASSERT_TRUE(fq_ops.empty());

    auto pool_ops = om.getOps("MaxPool");
    ASSERT_FALSE(pool_ops.empty());
    auto conv_ops = om.getOps("Conv");
    for(int i = 0; i < conv_ops.size(); i++) {
        auto quant_params = conv_ops[i]->get<mv::QuantizationParams>("quantParams");
        ASSERT_TRUE(quant_params.getScale()[0] != 1.0);
    }
}

TEST(fake_quantize_proc1, simple_case_no_fq_after_the_pass)
{
    mv::OpModel om("testModel");

    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");

    auto weightsData0 = generateRandomSequence(3 * 3 * 16, -0.3, 0.3, 42);
    auto weights0 = om.constant(weightsData0,{3,3,16,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4), {{0},{1.0},{},{}}, "Weights");

    std::vector<double> minData = mv::utils::generateSequence<double> (1, *std::min_element(weightsData0.begin(), weightsData0.end()), 0);
    std::vector<double> maxData = mv::utils::generateSequence<double> (1, *std::max_element(weightsData0.begin(), weightsData0.end()), 0);

    auto weights_min0 = om.constant(minData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto weights_max0 = om.constant(maxData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto weights_min1 = om.constant(minData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto weights_max1 = om.constant(maxData,{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));

    auto weights_fake_quant = om.fakeQuantize(weights0, weights_min0, weights_max0, weights_min1, weights_max1, 256);

    auto conv0 = om.conv(input0, weights_fake_quant, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("Default"), {{0},{1.0},{},{}}, "conv");

    om.output(conv0, mv::DType("Float16"));

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    auto fq_ops = om.getOps("FakeQuantize");
    ASSERT_FALSE(fq_ops.empty());
    mv::pass::PassRegistry::instance().find("FakeQuantize")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    fq_ops = om.getOps("FakeQuantize");
    ASSERT_TRUE(fq_ops.empty());
}

TEST(fake_quantize_proc, quantization_on_input)
{
    mv::OpModel om("testModel");

    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");

    auto fq_min0 = om.constant({0},{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto fq_max0 = om.constant({10},{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto fq_min1 = om.constant({0},{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));
    auto fq_max1 = om.constant({10},{1,1,1,1}, mv::DType("Float32"), mv::Order::getRowMajorID(4));

    auto input_fq = om.fakeQuantize(input0, fq_min0, fq_max0, fq_min1, fq_max1, 255);

    auto pool0 = om.maxPool(input_fq, {1, 1}, {1, 1}, {0, 0, 0, 0}, false, mv::DType("Default"), {{0},{1.0},{},{}}, "PoolLayer");

    om.output(pool0, mv::DType("Float16"));

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    auto pass = mv::pass::PassRegistry::instance().find("FakeQuantize");
    ASSERT_TRUE(pass != nullptr);
    pass->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    auto fq_ops = om.getOps("FakeQuantize");
    ASSERT_TRUE(fq_ops.empty());

    auto inputs = om.getOps("Input");
    for(int i = 0; i < inputs.size(); i++) {
        auto quant_params = inputs[i]->get<mv::QuantizationParams>("quantParams");
        ASSERT_TRUE(quant_params.getScale()[0] != 1.0);
    }
}