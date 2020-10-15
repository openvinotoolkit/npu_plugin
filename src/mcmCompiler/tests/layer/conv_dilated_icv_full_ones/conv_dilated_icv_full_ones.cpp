#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>
#include <include/mcm/op_model.hpp>

#include <limits>
int main()
{
    mv::CompilationUnit unit("ConvDilation");
    mv::OpModel& om = unit.model();

    static const auto inf = std::numeric_limits<double>::infinity();

    auto data_0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{0},{1},{0},{1}}, "input");

    //Kernel 3x3 - 16 input channel ; 16 output channels
    // 1 1 1
    // 1 1 1
    // 1 1 1
    // 1 1 1

    mv::Shape kernel = mv::Shape({3,3,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 255);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-1.000000000000000},{1.000000000000000}}, "weights_conv");

    // The dilation factor is 2
    // The padding is 2,2,2,2
    // This is SAME padding notation
    // Therefore subconvs padding will be 1,1,1,1 (as in slide 3 of design)

    auto conv0 = om.conv(data_0, weights0, {1, 1}, {2, 2, 2, 2}, 2, 1,  mv::DType("UInt8"),{{0},{0.5647058823529412},{0},{144},{0},{1}} , "conv");
    std::vector<int64_t> biasData0 = mv::utils::generateSequence<int64_t> (16, 0, 0);
    auto bias0 = om.constantInt(biasData0, {16}, mv::DType("Int32"), mv::Order::getRowMajorID(1), {{0}, {std::pow(0.00392156862745098,2)}, {}, {}});
    auto bias = om.bias(conv0, bias0, mv::DType("UInt8"), {{0}, {0.000000768935}, {}, {}});
    om.output(bias,mv::DType("UInt8"), {{},{},{},{}});

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
