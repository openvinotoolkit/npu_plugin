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

    auto data_0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{127},{0.007874016},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input");

    //Kernel 3x3 - 16 input channel ; 16 output channels
    // 1 1 1 
    // 1 1 1 
    // 1 1 1 
    // 1 1 1 
    
    uint8_t zeroPointWt =8;
    mv::Shape kernel = mv::Shape({3,3,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 1);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{zeroPointWt},{0.125},{-1.000000000000000},{1.000000000000000}}, "weights_conv");

    //the 2 is dilation factor
    auto conv0 = om.conv(data_0, weights0, {1, 1}, {2, 2, 2, 2}, 2, 1,  mv::DType("UInt8"),{{32},{4},{-inf},{inf},{0},{1}} , "conv");
    //add identity maxpool so concat will happen using storage elements

    auto identity_maxPool = om.maxPool(conv0, {1,1}, {1,1}, {0,0,0,0}, true, mv::DType("UInt8"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "identity_maxpool_0");

    om.output(identity_maxPool);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
