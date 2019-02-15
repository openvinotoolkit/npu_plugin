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

    // Define the primary compilation unit
    mv::CompilationUnit unit("Test");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel &cm = unit.model();

    /*Create computation model*/
    auto input = cm.input({224, 224, 3}, mv::DType("Float16"),  mv::Order("HWC"));

    /*Convolution*/
    mv::Shape kernelShape = {7, 7, 3, 64};
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());

    auto weights = cm.constant(weightsData, kernelShape, mv::DType("Float16"),  mv::Order("HWCN"));
    std::array<unsigned short, 2> stride = {2, 2};
    std::array<unsigned short, 4> padding = {3, 3, 3, 3};
    auto conv = cm.conv(input, weights, stride, padding, 1);

    /*convoultion bias*/
    mv::Shape convBiasShape = {64};
    std::vector<double> convBiasData = mv::utils::generateSequence<double>(convBiasShape.totalSize());
    auto convBiasTensor = cm.constant(convBiasData, convBiasShape, mv::DType("Float16"),  mv::Order("W"));
    auto convBias = cm.bias(conv,convBiasTensor);

    /*Scale*/
    std::vector<double> scaleData = mv::utils::generateSequence<double>(conv->get<mv::Shape>("shape")[2]);
    auto scaleTensor = cm.constant(scaleData, {conv->get<mv::Shape>("shape")[2]}, mv::DType("Float16"), mv::Order("W"));
    auto scale = cm.scale(convBias,scaleTensor);

    /*Scale bias*/
    mv::Shape scaleBiasShape = {64};
    std::vector<double> scaleBiasData = mv::utils::generateSequence<double>(scaleBiasShape.totalSize());
    auto scaleBiasTensor = cm.constant(scaleBiasData, scaleBiasShape, mv::DType("Float16"), mv::Order("W"));
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

    unit.loadCompilationDescriptor(mv::Target::ma2480);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string outputName = "prototxt";
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

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // compDesc.setPassArg("GenerateCaffe", "outputPrototxt", std::string("cppExampleprototxt.prototxt"));
    // compDesc.setPassArg("GenerateCaffe", "outputCaffeModel", std::string("cppExampleweights.caffemodel"));

    // Initialize compilation
    unit.initialize();

    // Run all passes
    unit.run();

    return 0;
}
