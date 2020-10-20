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

    auto data_0 = om.input("input", {32,32,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/);
    data_0->setQuantParams({{127},{0.007874016},{-1.000000000000000},{1.000000000000000},{0},{1}});

    //Kernel 3x3 - 16 input channel ; 16 output channels
    // 1 1 1
    // 1 1 1
    // 1 1 1
    // 1 1 1

    uint8_t zeroPointWt =8;
    mv::Shape kernel = mv::Shape({3,3,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 1);

    auto weights0 = om.constantInt("weights_conv", weightsData0, kernel, mv::DType("UInt8"), mv::Order("NCHW"));
    weights0->setQuantParams({{zeroPointWt},{0.125},{-1.000000000000000},{1.000000000000000}});

    // The dilation factor is 2
    // The padding is 2,2,2,2
    // This is SAME padding notation
    // Therefore subconvs padding will be 1,1,1,1 (as in slide 3 of design)

    auto conv0 = om.conv("conv", data_0, weights0, {1, 1}, {3, 3, 3, 3}, 3, 1);
    conv0->setQuantParams({{32},{4},{-inf},{inf},{0},{1}});

    // Identidy conv - this should not change output of dilated conv
    // Output of dilated Conv is all 3f's

    mv::Shape kernel1 = mv::Shape({3,3,16,16});
    std::vector<int64_t> weightsData1(kernel1.totalSize(), 0);

    for(unsigned i = 64; i < kernel1.totalSize(); i+=143)
        weightsData1[i] = 1;

    auto weights1 = om.constantInt("weights_conv1", weightsData1, kernel1, mv::DType("UInt8"), mv::Order("NHWC"));
    weights1->setQuantParams({{0},{1},{-1.000000000000000},{1.000000000000000}});
    auto conv1 = om.conv( "conv1", conv0, weights1,{1, 1}, {1, 1, 1, 1}, 1, 1);
    conv1->setQuantParams({{32},{4},{-inf},{inf},{0},{1}});

    om.output("", conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
