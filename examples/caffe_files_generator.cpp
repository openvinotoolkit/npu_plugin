/**
 * @brief Example presenting generation of Caffe prototxt and CaffeModel files
 * 
 * In this example a model is composed using MCMCompiler's Composition API. Then
 * the compilation is for target MA2480 is initialized and compilation passes scheduled by 
 * target descriptor are executed. Included GenerateCaffe pass will generate Caffe files.
 * 
 */

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);

    // Define the primary compilation unit
    mv::CompilationUnit unit("Test");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel &cm = unit.model();

    /*Create computation model*/
    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16,  mv::Order("HWC"));

    /*Convolution*/
    mv::Shape kernelShape = {7, 7, 3, 64};
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());

    auto weights = cm.constant(weightsData, kernelShape, mv::DTypeType::Float16,  mv::Order("HWCN"));
    std::array<unsigned short, 2> stride = {2, 2};
    std::array<unsigned short, 4> padding = {3, 3, 3, 3};
    auto conv = cm.conv(input, weights, stride, padding);

    /*convoultion bias*/
    mv::Shape convBiasShape = {64};
    std::vector<double> convBiasData = mv::utils::generateSequence<double>(convBiasShape.totalSize());
    auto convBiasTensor = cm.constant(convBiasData, convBiasShape, mv::DTypeType::Float16,  mv::Order("W"));
    auto convBias = cm.bias(conv,convBiasTensor);

    /*Scale*/
    std::vector<double> scaleData = mv::utils::generateSequence<double>(conv->get<mv::Shape>("shape")[2]);
    auto scaleTensor = cm.constant(scaleData, {conv->get<mv::Shape>("shape")[2]}, mv::DTypeType::Float16, mv::Order("W"));
    auto scale = cm.scale(convBias,scaleTensor);

    /*Scale bias*/
    mv::Shape scaleBiasShape = {64};
    std::vector<double> scaleBiasData = mv::utils::generateSequence<double>(scaleBiasShape.totalSize());
    auto scaleBiasTensor = cm.constant(scaleBiasData, scaleBiasShape, mv::DTypeType::Float16, mv::Order("W"));
    auto scaleBias = cm.bias(scale,scaleBiasTensor);

    /*Max Pool*/
    auto pool = cm.maxPool(scaleBias, {3, 3}, {2, 2}, {1, 1, 1, 1});
    /*Relu*/
    auto relu = cm.relu(pool);

    /*Average Pool*/
    auto pool1 = cm.averagePool(relu, {3, 3}, {2, 2}, {1, 1, 1, 1});

    /*Softmax*/
    auto softmax = cm.softmax(pool1);

    cm.output(softmax);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("prototxt.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("prototext.blob");
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    // Initialize compilation
    unit.initialize();

    // Run all passes
    unit.run();

    return 0;
}
