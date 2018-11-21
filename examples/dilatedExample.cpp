#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    // Define the primary compilation unit
    mv::CompilationUnit unit("DilatedExample");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input({32, 32, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = cm.constant(weightsData, {3, 3, 3, 1}, mv::DTypeType::Float16, mv::Order("NCWH"));
    auto conv = cm.conv(input, weights1, {1, 1}, {1, 1, 1, 1}, 2);
    auto output = cm.output(conv);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480)){
        exit(1);
    }
    
    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    std::string outputName = "DilatedExample";
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
    
    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName);
    printReport(returnValue, std::cout);
    return 0;
}
