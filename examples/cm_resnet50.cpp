/**
 * @brief Example presenting composition and compilation of the ResNet50 CNN
 * 
 * In this example ResNet50 model is composed using MCMCompiler's Composition API. Then
 * the compilation for the target device MA2480 is initialized and compilation passes scheduled by 
 * the target descriptor are executed. Included GenerateDot pass will generate *.dot files
 * that visualize the computation model at the end of each accomplished compilation phase.
 * Included GenerateBlob pass will serialize the model to a binary deployable to the target device.
 * 
 * Notes:
 * - This implementation of ResNet50 uses fused batch norm representation - batch norm is expressed
 * as a sequence of scale and bias
 * - This implementation of ResNet50 is aligned with Caffe - batch norm is followed by scale and bias
 * - Weights and other model parameters are initialized as sequences of numbers starting with 0
 * 
 * @file cm_resnet50.cpp
 * @author Stanislaw Maciag
 * @date 2018-07-19
 */

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

/**
 * @brief Helper function creates a chain of conv2D and fused batchnorm attached to the selected input tensor
 * 
 * @param model Master compositional model
 * @param input Tensor that is an input data for the conv2D
 * @param kernelShape Shape of conv2D kernel
 * @param stride Stride of conv2D
 * @param padding Padding of conv2D
 * @return mv::Data::TensorIterator Iterator referencing the created batchnorm output 
 */
mv::Data::TensorIterator convBatchNormBlock(mv::CompositionalModel& model,
                                            mv::Data::TensorIterator input,
                                            mv::Shape kernelShape,
                                            std::array<unsigned short, 2> stride,
                                            std::array<unsigned short, 4> padding,
                                            const mv::QuantizationParams& quantParams = {{}, {}, {}, {}},
                                            const std::string& name = "")
{

    std::string weightsName = "";
    std::string biasParamName = "";

    if(name != "")
    {
        weightsName = "weights_" + name;
        biasParamName = "biasParam_" + name;
    }
    
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(kernelShape.totalSize());

    auto weights = model.constantInt(weightsData, kernelShape, mv::DType("Int8"), mv::Order("HWCN"), quantParams, weightsName);
    auto conv = model.conv(input, weights, stride, padding, 1, 1, quantParams, "conv_" + name);
    
    std::vector<int64_t> biasData = mv::utils::generateSequence<int64_t>(conv->getShape()[mv::IO_CHANNEL_DIMENSION]);
    auto biasParam = model.constantInt(biasData,
                                    {conv->getShape()[mv::IO_CHANNEL_DIMENSION]},
                                    mv::DType("Int8"),
                                    mv::Order("W"),
                                    quantParams,
                                    biasParamName);

    return model.bias(conv, biasParam, quantParams, "bias_" + name);

}

/**
 * @brief Helper function that attaches a residual block to the selected input tensor
 * @param model Master compositional model
 * @param input Tensor that is an input data for first stages of residual block
 * @param intermediateDepth Number of output channels for the first convolution of the branch 2
 * @return mv::Data::TensorIterator Iterator referencing the created residual block output
 */
mv::Data::TensorIterator residualBlock(mv::CompositionalModel& model,
                                        mv::Data::TensorIterator input,
                                        unsigned intermediateDepth,
                                        const mv::QuantizationParams& quantParams = {{}, {}, {}, {}},
                                        const std::string& name = "")
{

    auto inputShape = input->getShape();
    auto branch2a = convBatchNormBlock(model,
                                        input,
                                        {1, 1, inputShape[mv::IO_CHANNEL_DIMENSION], intermediateDepth},
                                        {1, 1},
                                        {0, 0, 0, 0},
                                        quantParams,
                                        "branc2a_" + name);
    branch2a = model.relu(branch2a, quantParams, name + "_branch2a_relu");

    auto branch2b = convBatchNormBlock(model,
                                        branch2a,
                                        {3, 3, intermediateDepth, intermediateDepth},
                                        {1, 1},
                                        {1, 1, 1, 1},
                                        quantParams,
                                        "branch2b_" + name);
    branch2b = model.relu(branch2b, quantParams, name + "_branch2b_relu");

    auto branch2c = convBatchNormBlock(model,
                                        branch2b,
                                        {1, 1, intermediateDepth, inputShape[mv::IO_CHANNEL_DIMENSION]},
                                        {1, 1},
                                        {0, 0, 0, 0},
                                        quantParams,
                                        "branch2c_" + name);

    auto res = model.add(input, branch2c, quantParams, name + "_add");
    return model.relu(res, quantParams, name + "_relu");

}

/**
 * @brief Helper function that attaches a residual block (with conv2d on branch b) to the selected input tensor
 * @param model Master compositional model
 * @param input Tensor that is an input data for first stages of residual block 
 * @param intermediateDepth Number of output channels for the first convolution of the branch 2
 * @param outputDepth Number of output channels of the block
 * @param stride Stride applied for the convolution in branch 1 and the first convolution in branch 2
 * @return mv::Data::TensorIterator Iterator referencing the created residual block output
 */
mv::Data::TensorIterator
residualConvBlock(mv::CompositionalModel& model,
                mv::Data::TensorIterator input,
                unsigned intermediateDepth,
                unsigned outputDepth,
                std::array<unsigned short, 2> stride,
                const mv::QuantizationParams& quantParams = {{}, {}, {}, {}},
                const std::string& name = "")
{

    auto inputShape = input->getShape();
    auto branch1 = convBatchNormBlock(model,
                                        input,
                                        {1, 1, inputShape[mv::IO_CHANNEL_DIMENSION], outputDepth},
                                        stride,
                                        {0, 0, 0, 0},
                                        quantParams,
                                        "branch1_" + name);

    auto branch2a = convBatchNormBlock(model,
                                        input,
                                        {1, 1, inputShape[mv::IO_CHANNEL_DIMENSION], intermediateDepth},
                                        stride,
                                        {0, 0, 0, 0},
                                        quantParams,
                                        "branch2a_" + name);

    branch2a = model.relu(branch2a, quantParams, name + "_branch2a_relu");

    auto branch2b = convBatchNormBlock(model,
                                        branch2a,
                                        {3, 3, intermediateDepth, intermediateDepth},
                                        {1, 1},
                                        {1, 1, 1, 1},
                                        quantParams,
                                        "branch2b_" + name);
    branch2b = model.relu(branch2b, quantParams, name + "_branch2b_relu");

    auto branch2c = convBatchNormBlock(model,
                                        branch2b,
                                        {1, 1, intermediateDepth, outputDepth},
                                        {1, 1},
                                        {0, 0, 0, 0},
                                        quantParams,
                                        "branch2c_" + name);

    auto res = model.add(branch1, branch2c, quantParams, name + "_add");
    return model.relu(res, quantParams, name + "_relu");

}

int main()
{

    mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);

    // Define the primary compilation unit
    mv::CompilationUnit unit("ResNet50");

    std::string descPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    std::ifstream compDescFile(descPath);

    // Obtain a compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    // Compose the model for ResNet50

    // Define/import quantization params from somewhere
    mv::QuantizationParams emptyQuantParams({}, {}, {}, {});

    auto input = cm.input({224, 224, 3, 1}, mv::DType("Int8"), mv::Order("NHWC"), emptyQuantParams, "input");
    auto conv1 = convBatchNormBlock(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3}, emptyQuantParams, "conv1");
    conv1 = cm.relu(conv1, emptyQuantParams, "relu1");
    auto pool1 = cm.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto res2a = residualConvBlock(cm, pool1, 64, 256, {1, 1}, emptyQuantParams, "res2a");
    auto res2b = residualBlock(cm, res2a, 64, emptyQuantParams, "res2b");
    auto res2c = residualBlock(cm, res2b, 64, emptyQuantParams, "res2c");
    auto res3a = residualConvBlock(cm, res2c, 128, 512, {2, 2}, emptyQuantParams, "res3a");
    auto res3b = residualBlock(cm, res3a, 128, emptyQuantParams, "res3b");
    auto res3c = residualBlock(cm, res3b, 128, emptyQuantParams, "res3c");
    auto res3d = residualBlock(cm, res3c, 128, emptyQuantParams, "res3d");
    auto res4a = residualConvBlock(cm, res3d, 256, 1024, {2, 2}, emptyQuantParams, "res4a");
    auto res4b = residualBlock(cm, res4a, 256, emptyQuantParams, "res4b");
    auto res4c = residualBlock(cm, res4b, 256, emptyQuantParams, "res4c");
    auto res4d = residualBlock(cm, res4c, 256, emptyQuantParams, "res4d");
    auto res4e = residualBlock(cm, res4d, 256, emptyQuantParams, "res4e");
    auto res4f = residualBlock(cm, res4e, 256, emptyQuantParams, "res4f");
    // auto res5a = residualConvBlock(cm, res4f, 512, 2048, {2, 2}, emptyQuantParams, "res5a");
    // auto res5b = residualBlock(cm, res5a, 512, emptyQuantParams, "res5b");
    // auto res5c = residualBlock(cm, res5b, 512, emptyQuantParams, "res5c");
    // auto pool5 = cm.averagePool(res5c, {7, 7}, {1, 1}, {0, 0, 0, 0});
    cm.output(res4f);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2490))
        exit(1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    // Initialize compilation 
    unit.initialize();
    // Run all passes
    auto result = unit.run();

    return 0;

}
