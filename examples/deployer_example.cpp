#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "include/mcm/utils/hardware_tests.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/deployer/configuration.hpp"
#include "include/mcm/utils/deployer/executor.hpp"

#include <iostream>
#include <fstream>

#ifdef RESNET18_TEST
mv::Data::TensorIterator convBatchNormBlock1(mv::CompositionalModel& model, mv::Data::TensorIterator input,  mv::Shape kernelShape, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding)
{
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = model.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::Order("NCHW"));
    auto conv = model.conv(input, weights, stride, padding, 1);

    // For debugging purpose weights are initialized as sequences of numbers, to be replaced with actual weights
    std::vector<double> meanData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    std::vector<double> varianceData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    std::vector<double> offsetData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    std::vector<double> scaleData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    auto bnmean = model.constant(meanData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::Order("W"));
    auto bnvariance = model.constant(varianceData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::Order("W"));
    auto bnoffset = model.constant(offsetData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::Order("W"));
    auto bnscale = model.constant(scaleData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::Order("W"));
    return model.batchNormalization(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
}

/**
 * @brief Helper function that attaches a residual block to the selected input tensor
 * @param model Master compositional model
 * @param input Tensor that is an input data for first stages of residual block
 * @return mv::Data::TensorIterator Iterator referencing the created residual block output
 */
mv::Data::TensorIterator residualBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input)
{

	auto inputShape = input->getShape();
	auto branch2a = convBatchNormBlock1(model, input, {3, 3, inputShape[2], inputShape[2]}, {1, 1}, {1, 1, 1, 1});
	branch2a = model.relu(branch2a);
	auto branch2b = convBatchNormBlock1(model, branch2a, {3, 3, inputShape[2], inputShape[2]}, {1, 1}, {1, 1, 1, 1});

	auto res = model.add(input, branch2b);
	return model.relu(res);

}

/**
 * @brief Helper function that attaches a residual block (with conv2d on branch b) to the selected input tensor
 * @param model Master compositional model
 * @param input Tensor that is an input data for first stages of residual block
 * @return mv::Data::TensorIterator Iterator referencing the created residual block output
 */
mv::Data::TensorIterator residualConvBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, unsigned outputDepth, std::array<unsigned short, 2> stride)
{

	auto inputShape = input->getShape();
	auto branch1 = convBatchNormBlock1(model, input, {1, 1, inputShape[2], outputDepth}, stride, {0, 0, 0, 0});
	auto branch2a = convBatchNormBlock1(model, input, {3, 3, inputShape[2], outputDepth}, {1, 1}, {1, 1, 1, 1});
	branch2a = model.relu(branch2a);
	auto branch2b = convBatchNormBlock1(model, branch2a, {3, 3, outputDepth, outputDepth}, stride, {1, 1, 1, 1});

	auto res = model.add(branch1, branch2b);
	return model.relu(res);

}
#endif
// Create both RAM and file blobs
int main()
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);

    //mv::Logger::logFilter({std::regex("OpModel")}, true);
#ifdef RESNET18_TEST
    // Define the primary compilation unit
    mv::CompilationUnit unit("ResNet18");

    // Obtain compositional model from the compilation unit
    mv::OpModel& cm = unit.model();

    // Compose the model for ResNet18
    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    auto conv1 = convBatchNormBlock1(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3});
    conv1 = cm.relu(conv1);
    auto pool1 = cm.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto res2a = residualConvBlock(cm, pool1, 64, {1, 1});
    auto res2b = residualBlock(cm, res2a);
    auto res3a = residualConvBlock(cm, res2b, 128, {2, 2});
    auto res3b = residualBlock(cm, res3a);
    auto res4a = residualConvBlock(cm, res3b, 256, {2, 2});
    auto res4b = residualBlock(cm, res4a);
    auto res5a = residualConvBlock(cm, res4b, 512, {2, 2});
    auto res5b = residualBlock(cm, res5a);
    auto pool5 = cm.averagePool(res5b, {7, 7}, {1, 1,}, {0, 0, 0, 0});
    std::vector<double> weightsData = mv::utils::generateSequence<double>(pool5->getShape().totalSize() * 1000u);
    auto weights = cm.constant(weightsData, {pool5->getShape().totalSize(), 1000}, mv::DTypeType::Float16, mv::Order("HW"));
    auto fc1000 = cm.fullyConnected(pool5, weights);
    auto softmax = cm.softmax(fc1000);
    cm.output(softmax);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480)){
        exit(1);
    }

    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("cm_resnet18.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string("resnet18.blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    // Initialize compilation
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);

    // Run all passes
    unit.run();

#else
 // Define the primary compilation unit
    mv::CompilationUnit unit("gold11Model");
    mv::OpModel& cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = cm.input({64, 64, 3}, mv::DTypeType::Float16, mv::Order("WHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = cm.constant(weightsData61, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    std::vector<double> biasesData = { 64444.0 };
    auto biases = cm.constant(biasesData, {1}, mv::DTypeType::Float16, mv::Order("W"), "biases");
    auto bias1 = cm.bias(convIt61, biases);
    // define first maxpool
    auto maxpoolIt61 = cm.maxPool(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt62 = cm.constant(weightsData62, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt62 = cm.conv(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0}, 1);

    // define scale
    std::vector<double> scalesData = { 6550.0 };
    auto scales = cm.constant(scalesData, {1}, mv::DTypeType::Float16, mv::Order("W"), "scales");
    auto scaleIt62 = cm.scale(convIt62, scales);
    // define output
    auto outIt6 = cm.output(scaleIt62);

    std::string blobName = "test_scale_11.blob";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = blobName;
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = true;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("blob_eltwise_multiply.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();
#endif
    //Create Configuration
    mv::Configuration config(cm.getBinaryBuffer());
    mv::Executor exec(config);
    mv::Tensor res = exec.execute();
    std::cout << "res Order " << res.getOrder().toString() << std::endl;
    std::cout << "res Shape " << res.getShape().toString() << std::endl;
    std::cout << "ndims " << res.getShape().ndims() << std::endl;
    std::cout << "totalSize " << res.getShape().totalSize() << std::endl;
    //for (unsigned int i=0; i < res.getShape().totalSize(); i++)
    //    if (res(i) != 0)
    //        std::cout << "res[" << i << "] = " << res(i) << std::endl;
}