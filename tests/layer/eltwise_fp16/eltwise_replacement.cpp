#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("EltwiseReplacement");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({16,16,32,1});

    // Quantize zeropoints & scales
    auto in_qp0 = mv::QuantizationParams({{0}, {0.003921568627451}, {},{}});
    auto in_qp1 = mv::QuantizationParams({{0}, {0.003921568627451 * 2}, {},{}});
    auto out_qp = mv::QuantizationParams({{0}, {1.0}, {},{}});

    // Input
    auto input0 = om.input(inputShape, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{0.003921568627451},{},{}}, "input0");

    // Identity MaxPools
    auto identity_maxPool0 = om.maxPool(input0, {1,1}, {1,1}, {0,0,0,0}, true, "", "floor", mv::DType("UInt8"), in_qp0, "identity_maxpool0");
    auto identity_maxPool1 = om.maxPool(input0, {1,1}, {1,1}, {0,0,0,0}, true, "", "floor", mv::DType("UInt8"), in_qp1, "identity_maxpool1");

    // Eltwise, to be replaced with eltwiseFP16 in replacement pass
    auto eltwise0 = om.eltwise({identity_maxPool0, identity_maxPool1}, "Add", mv::DType("UInt8"), out_qp, "eltwise0");

    // Re-quantize output
    auto outputMaxPool0 = om.maxPool(eltwise0, {1,1}, {1,1}, {0,0,0,0}, true, "", "floor", mv::DType("UInt8"), out_qp, "requantize_output");

    // Output
    auto output0 = om.output(outputMaxPool0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_SC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
