#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("FullyConnected");
    mv::CompositionalModel& test_cm = unit.model();
    auto input = test_cm.input({8, 8, 16}, mv::DTypeType::Float16, mv::Order("CHW"));

    std::vector<double> weightsData = mv::utils::generateSequence<double>(input->getShape().totalSize() * 100u);
    auto weights1 = test_cm.constant(weightsData, {input->getShape().totalSize(), 100}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(2)));
    auto fullyConnected = test_cm.fullyConnected(input, weights1);
    auto output = test_cm.output(fullyConnected);

    std::string outputName = "FullyConnected";
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

    unit.passManager().disablePass(mv::PassGenre::Adaptation, "FullyConnectedAsConv2D");

    auto returnValue = mv::HWTest(unit, outputName, true);
    printReport(returnValue, std::cout);
}
