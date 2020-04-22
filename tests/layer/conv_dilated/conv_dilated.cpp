#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>
#include <include/mcm/op_model.hpp>
//#include "include/mcm/compiler/compilation_unit.hpp"
//#include "templateExampleNew.data.inc"

#include <limits>
int main()
{
    mv::CompilationUnit unit("ConvReluModel");
    mv::OpModel& om = unit.model();

    //static const auto inf = std::numeric_limits<double>::infinity();

    auto data_0 = om.input({16,16,1,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{127},{0.007874016},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input");

    //Kernel 3x3 
    //1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0
    // 1 1 1
    // 1 1 1
    // 1 1 1
    std::vector<int64_t> weightsData0 = {   1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1};

    auto weights0 = om.constantInt(weightsData0,{3,3,1,16}, mv::DType("UInt8"), mv::Order("NCHW"), {{145},{0.001586804166436},{-0.230153873562813},{0.174481183290482}}, "weights_conv");

    auto conv0 = om.conv(data_0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{1},{-65535},{65535},{0},{1}} , "conv");

    //the 2 is dilation factor
    //auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 2, 1,  mv::DType("UInt8"),{{0},{1},{},{}} , "conv");
    //dilation 1
    //auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{1},{},{}} , "conv");
    om.output(conv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
