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

    auto input = cm.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = cm.constant(weightsData, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = cm.conv(input, weights1, {1, 1}, {1, 1, 1, 1}, 2);
    auto output = cm.output(conv);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480)){
        exit(1);
    }
    

    unit.loadDefaultCompilationDescriptor();
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string outputName = "DilatedExample";
    mv::Attribute blobNameAttr(outputName + ".blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobNameAttr);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    // NOTE: GenerateDot is not applicable for release version. Use debug compilation
    // descriptor if needed.
    // compDesc.setPassArg("GenerateDot", "output", std::string(outputName + ".dot"));
    // compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    // compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    // compDesc.setPassArg("GenerateDot", "html", true);

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", false);

    // compDesc.setPassArg("GenerateCaffe", "outputPrototxt", std::string(outputName + ".prototxt"));
    // compDesc.setPassArg("GenerateCaffe", "outputCaffeModel", std::string(outputName + ".caffemodel"));

    unit.initialize();

    auto returnValue = mv::HWTest(unit, outputName);
    printReport(returnValue, std::cout);
    return 0;
}
