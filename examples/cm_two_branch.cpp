#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);

    // Define the primary compilation unit
    mv::CompilationUnit unit("cm_two_branch");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input({24, 24, 20}, mv::DTypeType::Float16, mv::Order("CHW"));
    auto pool1It = cm.maxPool(input, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool2It = cm.maxPool(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool3It = cm.maxPool(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});

    auto concat1It = cm.add(pool3It, pool2It);
    auto pool4It = cm.maxPool(concat1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    cm.output(pool4It);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("cm_twobranch.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string("cm_twobranch.blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;
    

    // Initialize compilation
    unit.initialize();
    
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);
    unit.passManager().disablePass(mv::PassGenre::Adaptation, "GenerateCaffe");

    // Run all passes
    unit.run();

    system("dot -Tsvg cm_two_branch.dot -o cm_two_branch.svg");
    system("dot -Tsvg cm_two_branch_adapt.dot -o cm_two_branch_adapt.svg");
    system("dot -Tsvg cm_two_branch_final.dot -o cm_two_branch_final.svg");
    return 0;

}
