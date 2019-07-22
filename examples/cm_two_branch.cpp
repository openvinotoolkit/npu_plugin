#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("cm_two_branch");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input({24, 24, 20}, mv::DType("Float16"), mv::Order("CHW"));
    auto pool1It = cm.maxPool(input, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool2It = cm.maxPool(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool3It = cm.maxPool(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});

    auto concat1It = cm.add({pool3It, pool2It});
    auto pool4It = cm.maxPool(concat1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    cm.output(pool4It);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    unit.loadCompilationDescriptor(mv::Target::ma2480);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "cm_two_branch.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    // NOTE: GenerateDot is not applicable for release version. Use debug compilation
    // descriptor if needed.
    // compDesc.setPassArg("GenerateDot", "output", std::string("cm_two_branch.dot"));
    // compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    // compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    // compDesc.setPassArg("GenerateDot", "html", true);

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // compDesc.setPassArg("GenerateCaffe", "outputPrototxt", std::string("cppExampleprototxt.prototxt"));
    // compDesc.setPassArg("GenerateCaffe", "outputCaffeModel", std::string("cppExampleweights.caffemodel"));


    // Initialize compilation
    unit.initialize();
    
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);
    //unit.passManager().disablePass(mv::PassGenre::Adaptation, "GenerateCaffe");

    // Run all passes
    unit.run();

    //system("dot -Tsvg cm_two_branch.dot -o cm_two_branch.svg");
    //system("dot -Tsvg cm_two_branch_adapt.dot -o cm_two_branch_adapt.svg");
    //system("dot -Tsvg cm_two_branch_final.dot -o cm_two_branch_final.svg");
    return 0;

}
