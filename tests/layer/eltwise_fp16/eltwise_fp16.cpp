#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("EltwiseReplacement");
    mv::OpModel& om = unit.model();

    // Define parameters
    auto out_qp = mv::QuantizationParams({{0}, {1.0}, {},{}});
    std::vector<uint16_t> weightsData(7*7*1024);

    //Input
    auto input0 = om.input({7,7,1024,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input0");

    //Load weights from file
    std::string  weights_filename(mv::utils::projectRootPath() + "/tests/layer/eltwise_fp16/eltwise_fp16.in2");
    std::ifstream w_file;
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 7*7*1024 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(7*7*1024);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    auto weights0 = om.constantInt(weightsData_converted,{7,7,1024,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.1524552064656746e-05},{},{}}, "weights0");

    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    inputs.push_back(weights0);

    // Eltwise, to be replaced with eltwiseFP16 in replacement pass
    auto eltwise0 = om.eltwise(inputs, "Add", mv::DType("Float16"), {{0},{1.0},{},{}}, "eltwise0");

    // Output
    auto output0 = om.output(eltwise0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}