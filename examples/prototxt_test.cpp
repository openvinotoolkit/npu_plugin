#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{
    mv::Logger::setVerboseLevel(mv::Logger::VerboseLevel::VerboseDebug);

    // Define the primary compilation unit
    mv::CompilationUnit unit("Test");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel &cm = unit.model();

    /*Create computation model*/
    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);

    /*Convolution*/
    mv::Shape kernelShape = {3, 3, 3, 4};
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = cm.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    std::array<unsigned short, 2> stride = {2, 2};
    std::array<unsigned short, 4> padding = {3, 3, 3, 3};
    auto conv = cm.conv2D(input, weights, stride, padding);

    /*Bias*/
    std::vector<double> biasData = mv::utils::generateSequence<double>(conv->get<mv::Shape>("shape")[2]);
    auto biasTensor = cm.constant(biasData, {conv->get<mv::Shape>("shape")[2]}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto bias = cm.bias(conv,biasTensor);

    /*Scale*/
    std::vector<double> scaleData = mv::utils::generateSequence<double>(conv->get<mv::Shape>("shape")[2]);
    auto scaleTensor = cm.constant(scaleData, {conv->get<mv::Shape>("shape")[2]}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto scale = cm.scale(bias,scaleTensor);
 
    /*Pool*/
    auto pool = cm.maxpool2D(scale, {3, 3}, {2, 2}, {1, 1, 1, 1});
    
    /*Relu*/
    auto relu = cm.relu(pool);

    /*Softmax*/
    auto softmax = cm.softmax(relu);
    cm.output(softmax);

    mv::OpModel &opModel = dynamic_cast<mv::OpModel &>(cm);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("prototxt.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("prototext.blob");
    unit.compilationDescriptor()["GenerateProto"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateProto"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    // Initialize compilation
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Adaptation);
    //unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);

    // Run all passes
    unit.run();

    system("dot -Tsvg prototxt.dot -o protoxt.svg");
    //system("dot -Tsvg prototxt_adapt.dot -o prototxt_adapt.svg");
    //system("dot -Tsvg prototxt_final.dot -o prototxt_final.svg");
    

    return 0;
}
