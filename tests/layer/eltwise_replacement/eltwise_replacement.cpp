#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("EltwiseReplacement");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({7,7,1024,1});

    // Quantize zeropoints & scales
    auto in_qp0 = mv::QuantizationParams({{0}, {1.0}, {},{}});
    auto in_qp1 = mv::QuantizationParams({{0}, {0.5}, {},{}});
    auto out_qp = mv::QuantizationParams({{0}, {2.0}, {},{}});
    auto identity_qp = mv::QuantizationParams({{0}, {1.0}, {},{}});

    // Input
    auto input0 = om.input(inputShape, mv::DType("UInt8"), mv::Order::getZMajorID(4), identity_qp, "input0");

    // MaxPools, with different quantParams from each other
    // Note: scaling_maxPool multiplies input by 2, then succeeding de-quantize op divides by 2
    auto identity_maxPool_in0 = om.maxPool(input0, {1,1}, {1,1}, {0,0,0,0}, true, "", "floor", mv::DType("UInt8"), in_qp0, "identity_maxpool_in0");
    auto scaling_maxPool_in1 = om.maxPool(input0, {1,1}, {1,1}, {0,0,0,0}, true, "", "floor", mv::DType("UInt8"), in_qp1, "scaling_maxpool_in1");

    // Eltwise, to be replaced with SW eltwise in replacement pass
    auto eltwise0 = om.eltwise({identity_maxPool_in0, scaling_maxPool_in1}, "Add", mv::DType("Float16"), identity_qp, "eltwise0");

    // Identity MaxPool
    //auto identity_maxPool_out0 = om.maxPool(eltwise0, {1,1}, {1,1}, {0,0,0,0}, true, "", "floor", mv::DType("Default"), out_qp, "identity_maxpool_out0");
    auto quantize_out0 = om.quantize(eltwise0, mv::DType("UInt8"), out_qp, "quantize_out0");

    // Output
    auto output0 = om.output(quantize_out0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
