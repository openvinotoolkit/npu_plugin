//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "build/meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MCM_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({112,112,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input:0#1");

    std::string weightsPath = path + "/example/normalize_only/normalize.weights";
    std::vector<double> scaleWeights0;
    double weight;
    std::fstream fs;
    fs.open(weightsPath, std::fstream::in);
    while( fs >> weight ) {
        scaleWeights0.push_back(weight);
    }
    fs.close();

    auto scales0 = om.constant(scaleWeights0,{1,1,32,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.1524552064656746e-05},{-inf},{inf}}, "scale_weights#0");
    
    double eps = 0.001; 
    auto normalize0 = om.normalize(input0, scales0, eps, 0, 0, mv::DType("Float16"));

    om.output(normalize0);

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
