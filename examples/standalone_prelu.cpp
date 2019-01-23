#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("prelu");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> data = mv::utils::generateSequence<double>(3);
    auto slope = om.constant(data, {3}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)));
    auto prelu = om.prelu(input, slope);
    auto output = om.output(prelu);

    std::string outputName = "test_prelu";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string(outputName+".blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName+".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string(outputName + ".prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string(outputName + ".caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName);
    printReport(returnValue, std::cout);
}
