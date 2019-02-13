#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

//NOTE: Does not work just for input size mismatch, op is actually ok
int main()
{
    mv::CompilationUnit unit("FullyConnected");
    mv::CompositionalModel& test_cm = unit.model();
    auto input = test_cm.input({8, 8, 16}, mv::DType("Float16"), mv::Order("CHW"));

    std::vector<double> weightsData = mv::utils::generateSequence<double>(input->getShape().totalSize() * 100u);
    auto weights1 = test_cm.constant(weightsData, {input->getShape().totalSize(), 100}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(2)));
    auto fullyConnected = test_cm.fullyConnected(input, weights1);
    auto output = test_cm.output(fullyConnected);

    unit.loadDefaultCompilationDescriptor();
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string outputName = "FullyConnected";
    mv::Attribute blobNameAttr(outputName + ".blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobNameAttr);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    compDesc.setPassArg("GenerateDot", "output", std::string(outputName + ".dot"));
    compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    compDesc.setPassArg("GenerateDot", "html", true);

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    compDesc.remove("adapt", "FullyConnectedAsConv2D");

    // compDesc.setPassArg("GenerateCaffe", "outputPrototxt", std::string(outputName + ".prototxt"));
    // compDesc.setPassArg("GenerateCaffe", "outputCaffeModel", std::string(outputName + ".caffemodel"));

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName, true);
    printReport(returnValue, std::cout);
}
