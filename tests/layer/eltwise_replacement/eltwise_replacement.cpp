#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("EltwiseReplacement");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({7,7,1024,1});

    // Quantize zeropoints & scales
    auto in_qp0 = mv::QuantizationParams({{0}, {0.00390625}, {},{}});
    auto in_qp1 = mv::QuantizationParams({{0}, {0.00390625 * 2}, {},{}});
    auto out_qp = mv::QuantizationParams({{0}, {0.00390625 * 2}, {},{}});

    // Input
    auto input0 = om.input(inputShape, mv::DType("UInt8"), mv::Order::getZMajorID(4), in_qp0, "input0");

    // De-quantize inputs
    auto quantize_in0 = om.quantize(input0, mv::DType("Float16"), in_qp0, "quantize_in0");
    auto quantize_in1 = om.quantize(input0, mv::DType("Float16"), in_qp1, "quantize_in1");

    // Eltwise, to be replaced with SW eltwise in replacement pass
    auto eltwise0 = om.eltwise({quantize_in0, quantize_in1}, "Add", mv::DType("Default"), out_qp, "eltwise0");

    // Re-quantize output
    auto quantize_out0 = om.quantize(eltwise0, mv::DType("UInt8"), out_qp, "quantize_out0");

    // Output
    auto output0 = om.output(quantize_out0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
