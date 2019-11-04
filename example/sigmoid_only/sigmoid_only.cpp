#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include "iostream"
#include "fstream"

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({5,1,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1},{-inf},{inf}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (8, 1, 0);
    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (8, -1, 0);
    weightsData0.insert(weightsData0.end(), weightsData1.begin(), weightsData1.end());

    auto weights0 = om.constantInt(weightsData0,{1,1,16,1}, mv::DType("Int8"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}});
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("Float16"),{{0},{1},{-inf},{inf}} , "conv");
    auto sigmoid0 = om.sigmoid(conv0, mv::DType("Float16"), {{0},{1},{-inf},{inf}}, "sigmoid0");
    om.output(sigmoid0);

    std::string path = std::getenv("MCM_HOME");
    std::string compDescPath = path + "/config/compilation/release_kmb_MC-Prefetch2.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}


