#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    auto input1 = test_cm.input({225, 225, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(input1->getShape().totalSize() * 1000u);
    auto weights = test_cm.constant(weightsData, {input1->getShape().totalSize(), 1000}, mv::DType("Float16"), mv::Order("HW"));
    auto fc1000 = test_cm.fullyConnected(input1, weights);
    auto output = test_cm.output(fc1000);

    std::string outputName = "wddm_fullyconnected1";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = outputName + ".blob";
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName + ".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string(outputName + ".prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string(outputName + ".caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = false;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName, true);
    printReport(returnValue, std::cout);
}
