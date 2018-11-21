#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("DepthwiseConvolution");
    mv::CompositionalModel& test_cm = unit.model();

    auto input = test_cm.input({32, 32, 1}, mv::DTypeType::Float16, mv::Order("CHW"));
    std::vector<double> weightsData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    auto weights1 = test_cm.constant(weightsData, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(4)));
    auto conv = test_cm.depthwiseConv(input, weights1, {4, 4}, {1, 1, 1, 1});
    auto output = test_cm.output(conv);

    std::string outputName = "DepthwiseConvolution";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = outputName + ".blob";
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName + ".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string(outputName + ".prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string(outputName + ".caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName, true);
    printReport(returnValue, std::cout);
}
