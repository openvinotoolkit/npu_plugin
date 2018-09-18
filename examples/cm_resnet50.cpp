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
mv::Data::TensorIterator convBatchNormBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input,  mv::Shape kernelShape, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding)
{
    
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = model.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto conv = model.conv2D(input, weights, stride, padding);

    // For debugging purpose weights are initialized as sequences of numbers, to be replaced with actual weights
    std::vector<double> bnScaleData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    auto bnScaleParam = model.constant(bnScaleData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto bnScale = model.scale(conv, bnScaleParam);
    
    std::vector<double> bnOffsetData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    auto bnOffsetParam = model.constant(bnOffsetData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto bnOffset = model.bias(bnScale, bnOffsetParam);
    
    std::vector<double> scaleData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    auto scaleParam = model.constant(scaleData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto scale = model.scale(bnOffset, scaleParam);
    
    std::vector<double> biasData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    auto biasParam = model.constant(biasData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    return model.bias(scale, biasParam);

}

/**
 * @brief Helper function that attaches a residual block to the selected input tensor
 * @param model Master compositional model
 * @param input Tensor that is an input data for first stages of residual block
 * @param intermediateDepth Number of output channels for the first convolution of the branch 2
 * @return mv::Data::TensorIterator Iterator referencing the created residual block output
 */
mv::Data::TensorIterator residualBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, unsigned intermediateDepth)
{

    auto inputShape = input->getShape();
    auto branch2a = convBatchNormBlock(model, input, {1, 1, inputShape[2], intermediateDepth}, {1, 1}, {0, 0, 0, 0});
    branch2a = model.relu(branch2a);
    auto branch2b = convBatchNormBlock(model, branch2a, {3, 3, intermediateDepth, intermediateDepth}, {1, 1}, {1, 1, 1, 1});
    branch2b = model.relu(branch2b);
    auto branch2c = convBatchNormBlock(model, branch2b, {1, 1, intermediateDepth, inputShape[2]}, {1, 1}, {0, 0, 0, 0});

    auto res = model.add(input, branch2c);
    return model.relu(res);

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
mv::Data::TensorIterator residualConvBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, unsigned intermediateDepth,
    unsigned outputDepth, std::array<unsigned short, 2> stride)
{

    auto inputShape = input->getShape();
    auto branch1 = convBatchNormBlock(model, input, {1, 1, inputShape[2], outputDepth}, stride, {0, 0, 0, 0});
    auto branch2a = convBatchNormBlock(model, input, {1, 1, inputShape[2], intermediateDepth}, stride, {0, 0, 0, 0});
    branch2a = model.relu(branch2a);
    auto branch2b = convBatchNormBlock(model, branch2a, {3, 3, intermediateDepth, intermediateDepth}, {1, 1}, {1, 1, 1, 1});
    branch2b = model.relu(branch2b);
    auto branch2c = convBatchNormBlock(model, branch2b, {1, 1, intermediateDepth, outputDepth}, {1, 1}, {0, 0, 0, 0});

    auto res = model.add(branch1, branch2c);
    return model.relu(res);

}

int main()
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("ResNet50");

    // Obtain a compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    // Compose the model for ResNet50
    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto conv1 = convBatchNormBlock(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3});
    conv1 = cm.relu(conv1);
    auto pool1 = cm.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto res2a = residualConvBlock(cm, pool1, 64, 256, {1, 1});
    auto res2b = residualBlock(cm, res2a, 64);
    auto res2c = residualBlock(cm, res2b, 64);
    auto res3a = residualConvBlock(cm, res2c, 128, 512, {2, 2});
    auto res3b = residualBlock(cm, res3a, 128);
    auto res3c = residualBlock(cm, res3b, 128);
    auto res3d = residualBlock(cm, res3c, 128);
    auto res4a = residualConvBlock(cm, res3d, 256, 1024, {2, 2});
    auto res4b = residualBlock(cm, res4a, 256);
    auto res4c = residualBlock(cm, res4b, 256);
    auto res4d = residualBlock(cm, res4c, 256);
    auto res4e = residualBlock(cm, res4d, 256);
    auto res4f = residualBlock(cm, res4e, 256);
    auto res5a = residualConvBlock(cm, res4f, 512, 2048, {2, 2});
    auto res5b = residualBlock(cm, res5a, 512);
    auto res5c = residualBlock(cm, res5b, 512);
    auto pool5 = cm.avgpool2D(res5c, {7, 7}, {1, 1}, {0, 0, 0, 0});
    std::vector<double> weightsData = mv::utils::generateSequence<double>(pool5->getShape().totalSize() * 1000u);
    auto weights = cm.constant(weightsData, {pool5->getShape().totalSize(), 1000}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto fc1000 = cm.fullyConnected(pool5, weights);
    auto softmax = cm.softmax(fc1000);
    cm.output(softmax);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from the compilation unit
    // Output DOT - file name (base)
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("resnet50.dot");
    // Output DOT - scope of visualization - executable operations, data flow, control flow
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
    // Output DOT - content included in the visualization - full content
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    // Output DOT - HTML-like flag - enable HTML-like formatting
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    // Output BLOB - file name of the output binary
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("resnet50.blob");
    //unit.compilationDescriptor()["GenerateJSON"]["output"] = std::string("resnet50.json");

    // Initialize compilation 
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);

    // Run all passes
    auto result = unit.run();

    // Obtain generated binary size from the compilation output
    //std::cout << "BLOB size: " << result["passes"].last()["blobSize"].get<long long>() << std::endl;
    std::cout << result.stringifyPretty() << std::endl;

    // Uncomment for an easy generation of SVG images for DOT output files (requires dot package)
    //system("dot -Tsvg resnet50.dot -o resnet50.svg");
    //system("dot -Tsvg resnet50_adapt.dot -o resnet50_adapt.svg");
    //system("dot -Tsvg resnet50_final.dot -o resnet50_final.svg");

    return 0;

}
