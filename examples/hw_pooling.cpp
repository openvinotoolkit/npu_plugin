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
    auto pool1 = test_cm.maxPool(input1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto output = test_cm.output(pool1);

    unit.loadDefaultCompilationDescriptor();
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string outputName = "wddm_pool1";
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

    auto returnValue = mv::HWTest(unit, outputName, true);
    printReport(returnValue, std::cout);
}
