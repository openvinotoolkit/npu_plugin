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
    auto input0 = om.input({16,16,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input:0#1");

    std::string weightsPath = path + "/example/normalize_only/normalize.weights";

    std::vector<int64_t> scaleWeights0;

    std::fstream fs;
    fs.open(weightsPath, std::fstream::in);

    for(int i = 0; i < 32; ++i) {
        int16_t x;
        fs.read((char*)&x, 2);
        scaleWeights0.push_back(x);
    }
    fs.close();

    auto scales0 = om.constantInt(scaleWeights0,{1,1,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.1524552064656746e-05},{-inf},{inf}}, "scale_weights#0");
    
    double eps = 0.001; 
    auto normalize0 = om.normalize(input0, scales0, eps);

    om.output(normalize0);

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
