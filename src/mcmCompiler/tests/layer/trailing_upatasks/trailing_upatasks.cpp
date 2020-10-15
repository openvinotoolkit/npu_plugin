#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("TrailingUPATasks");
    mv::OpModel& om = unit.model();

    auto input_shape = mv::Shape({1,1,256,1});

    // Input
    auto input0 = om.input(input_shape, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input");

    // DPUTask
    std::string weights_filename(mv::utils::projectRootPath() + "/tests/layer/trailing_upatasks/trailing_upatasks.w");
    std::vector<int64_t> weightsData0 = mv::utils::readWeightsFromFile<int64_t>(weights_filename);
    auto weights0 = om.constantInt(weightsData0,{1,1,256,256}, mv::DType("Int8"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}});
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{1.0},{},{}} , "conv0");

    // UPATasks, non-trailing
    auto reshape0 = om.reshape(conv0, {8,8,4,1}, mv::DType("Float16"), {{0},{1.0},{},{}}, "reshape0");
    auto reshape1 = om.reshape(reshape0, {1,1,256,1}, mv::DType("Float16"), {{0},{1.0},{},{}}, "reshape1");

    // DPUTask
    auto pool0 = om.maxPool(reshape1, {1,1}, {1,1}, {0,0,0,0}, true, mv::DType("UInt8"), {{0},{1.0},{},{}}, "pool0");

    // UPATasks, trailing
    auto reshape2 = om.reshape(pool0, {8,8,4,1}, mv::DType("Float16"), {{0},{1.0},{},{}}, "reshape2");
    auto reshape3 = om.reshape(reshape2, {1,1,256,1}, mv::DType("Float16"), {{0},{1.0},{},{}}, "reshape3");

    // Output
    om.output(reshape3, mv::DType("UInt8"));

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
