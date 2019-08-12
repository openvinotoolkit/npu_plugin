#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("leaky_relu");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    double alpha = 0.1;
    auto leaky_relu = om.leakyRelu(input, alpha);
    auto output = om.output(leaky_relu);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2480.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string outputName = "test_leaky_relu";
    mv::Attribute blobNameAttr(outputName + ".blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobNameAttr);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    compDesc.setPassArg("GenerateDot", "output", std::string(outputName + ".dot"));
    compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    compDesc.setPassArg("GenerateDot", "html", true);

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // compDesc.setPassArg("GenerateCaffe", "outputPrototxt", std::string(outputName + ".prototxt"));
    // compDesc.setPassArg("GenerateCaffe", "outputCaffeModel", std::string(outputName + ".caffemodel"));

    unit.loadTargetDescriptor(mv::Target::ma2480);

    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName);
    printReport(returnValue, std::cout);
}
