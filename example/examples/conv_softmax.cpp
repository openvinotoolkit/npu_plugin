#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MCM_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({32,32,32,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (1*1*32*32);
    auto weights0 = om.constantInt(weightsData0,{1,1,32,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "weights0");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "conv0");

    auto softmax0 = om.softmax(conv0, std::string("C"), mv::DType("Float16"));
    om.output(softmax0);

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
