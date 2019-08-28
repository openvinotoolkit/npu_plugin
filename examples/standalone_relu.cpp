#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("relu");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto relu = om.relu(input);
    auto output = om.output(relu);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2480.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string outputName = "test_relu";
    mv::Attribute blobNameAttr(outputName + ".blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobNameAttr);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    compDesc.setPassArg("GenerateDot", "output", std::string("test_relu.dot"));
    compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    compDesc.setPassArg("GenerateDot", "html", true);

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // compDesc.setPassArg("GenerateCaffe", "outputPrototxt", std::string("test_relu.prototxt"));
    // compDesc.setPassArg("GenerateCaffe", "outputCaffeModel", std::string("test_relu.caffemodel"));

    unit.loadTargetDescriptor(mv::Target::ma2480);

    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName);
    printReport(returnValue, std::cout);
}
