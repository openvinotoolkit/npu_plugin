

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    //mv::Logger::logFilter({std::regex("OpModel")}, true);

    // Define the primary compilation unit
    mv::CompilationUnit unit("DilatedExample");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input({32, 32, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = cm.constant(weightsData, {3, 3, 3, 1}, mv::DTypeType::Float16, mv::Order("NCWH"));
    auto conv = cm.conv(input, weights1, {1, 1}, {1, 1, 1, 1}, 1);
    auto output = cm.output(conv);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480)){
        exit(1);
    }
    
    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("test.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string("dilation.blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;
    
    // Initialize compilation 
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);

    // Run all passes
    unit.run();

    system("dot -Tsvg test.dot -o test.svg");
    system("dot -Tsvg test_adapt.dot -o test_adapt.svg");
    system("dot -Tsvg test_final.dot -o test_final.svg");
    return 0;
}
