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

    auto data_0 = om.input({32,32,512,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{-inf},{inf},{0},{1}}, "input");


    mv::Shape kernel = mv::Shape({3,3,512,1024});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 0);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-inf},{inf}}, "weights_conv");


    auto conv0 = om.conv(data_0, weights0, {1, 1}, {6,6,6,6}, 6, 1,  mv::DType("UInt8"),{{0},{18504.282352941176},{-inf},{inf},{0},{1}} , "conv");

    om.output(conv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
