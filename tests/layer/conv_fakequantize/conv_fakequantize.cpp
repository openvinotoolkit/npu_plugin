#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>

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

int main()
{
    mv::CompilationUnit unit("ConvFakeQuantizeModel");
    mv::OpModel& om = unit.model();

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

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
