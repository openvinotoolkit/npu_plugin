#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("NormalizeModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({112,112,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input0");
    std::vector<uint16_t> weightsData(32);

    //Load weights tensor from file
    std::string  weights_filename(mv::utils::projectRootPath() + "/tests/layer/normalize/normalize.in2");
    std::ifstream w_file;
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 32 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(32);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    auto weights0 = om.constantInt(weightsData_converted,{1,1,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.1524552064656746e-05},{},{}}, "weights0");
    
    double eps = 0.001; 
    auto normalize0 = om.normalize(input0, weights0, eps, 0, 0, mv::DType("Float16"));

    om.output(normalize0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
