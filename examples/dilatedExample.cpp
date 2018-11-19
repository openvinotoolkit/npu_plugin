

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    //mv::Logger::logFilter({std::regex("OpModel")}, true);

    // Define the primary compilation unit
    mv::CompilationUnit unit("DilatedExample");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input({32, 32, 1}, mv::DTypeType::Float16, mv::Order("CHW"));
    std::vector<double> weightsData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    //, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
    //15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f});
    auto weights1 = cm.constant(weightsData, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(4)));
    auto conv = cm.convDilated(input, weights1, {1, 1}, {1, 1, 1, 1}, 4);
    auto output = cm.output(conv);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480)){
        exit(1);
    }
    
    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("cm_resnet18.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string("resnet18.blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;
    
    // Initialize compilation 
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);

    // Run all passes
    unit.run();

    //system("dot -Tsvg cm_resnet18.dot -o cm_resnet18.svg");
    //system("dot -Tsvg cm_resnet18_adapt.dot -o cm_resnet18_adapt.svg");
    //system("dot -Tsvg cm_resnet18_final.dot -o cm_resnet18_final.svg");
    return 0;
}
