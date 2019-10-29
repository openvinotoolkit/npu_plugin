        //This file is the parsed network which is created through python.
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

    double alpha = 1;

    auto input0 = om.input({1,1,1000,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");
    auto elu0 = om.elu(input0, alpha);
    om.output(elu0);
    std::string compDescPath = path + "/config/compilation/release_kmb_MC-Prefetch1-Sparse.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}

// elu(Data::TensorIterator data, const unsigned& alpha = 1, const std::string& name = "") = 0;
