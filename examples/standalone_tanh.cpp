#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("tanh");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto tanh = om.tanh(input);
    auto output = om.output(tanh);

    unit.loadDefaultCompilationDescriptor();
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string outputName = "test_tanh";
    std::string blobName = outputName + ".blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    compDesc.setPassArg("GenerateDot", "output", std::string("test_tanh.dot"));
    compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    compDesc.setPassArg("GenerateDot", "html", true);

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // compDesc.setPassArg("GenerateCaffe", "outputPrototxt", std::string("test_tanh.prototxt"));
    // compDesc.setPassArg("GenerateCaffe", "outputCaffeModel", std::string("test_tanh.caffemodel"));

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName);
    printReport(returnValue, std::cout);
}
