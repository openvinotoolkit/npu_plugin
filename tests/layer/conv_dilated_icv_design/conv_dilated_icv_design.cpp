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

    auto data_0 = om.input({16,16,1,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{127},{0.007874016},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input");

    //Kernel 4x4 - 1 input channel ; 16 output channels
    // 1 1 1 1
    // 1 1 1 1
    // 1 1 1 1
    // 1 1 1 1
    
    uint8_t zeroPointWt =8;
    mv::Shape kernel = mv::Shape({4,4,1,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), zeroPointWt+zeroPointWt);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{zeroPointWt},{0.125},{-1.000000000000000},{1.000000000000000}}, "weights_conv");

    //the 2 is dilation factor
    auto conv0 = om.conv(data_0, weights0, {1, 1}, {1, 1, 1, 1}, 2, 1,  mv::DType("UInt8"),{{32},{4},{-inf},{inf},{0},{1}} , "conv");
    om.output(conv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
