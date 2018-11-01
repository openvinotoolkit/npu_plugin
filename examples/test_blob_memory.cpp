/**
 * @brief Example presenting composition of the ResNet18 network and its compilation
 * 
 * In this example ResNet18 model is composed using MCMCompiler's Composition API. Then
 * the compilation is for target MA2480 is initialized and compilation passes scheduled by 
 * target descriptor are executed. Included GenerateDot pass will generate *.dot files
 * that visualize the computation model at the end of each accomplished compilation phase.
 * 
 * @file cm_resnet18.cpp
 * @author Stanislaw Maciag
 * @date 2018-07-19
 */

#define COMPOSITIONAL_MODEL_RECORDER

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

/**
 * @brief Helper function creates a chain of conv2D and batchnorm attached to the selected input tensor
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
	auto weights = model.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
	auto conv = model.conv2D(input, weights, stride, padding);

	// For debugging purpose weights are initialized as sequences of numbers, to be replaced with actual weights
	std::vector<double> meanData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
	std::vector<double> varianceData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
	std::vector<double> offsetData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
	std::vector<double> scaleData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
	auto bnmean = model.constant(meanData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
	auto bnvariance = model.constant(varianceData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
	auto bnoffset = model.constant(offsetData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
	auto bnscale = model.constant(scaleData, {conv->getShape()[-1]}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
	return model.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);

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
	auto branch2a = convBatchNormBlock(model, input, {3, 3, inputShape[2], inputShape[2]}, {1, 1}, {1, 1, 1, 1});
	branch2a = model.relu(branch2a);
	auto branch2b = convBatchNormBlock(model, branch2a, {3, 3, inputShape[2], inputShape[2]}, {1, 1}, {1, 1, 1, 1});

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
	auto branch1 = convBatchNormBlock(model, input, {1, 1, inputShape[2], outputDepth}, stride, {0, 0, 0, 0});
	auto branch2a = convBatchNormBlock(model, input, {3, 3, inputShape[2], outputDepth}, {1, 1}, {1, 1, 1, 1});
	branch2a = model.relu(branch2a);
	auto branch2b = convBatchNormBlock(model, branch2a, {3, 3, outputDepth, outputDepth}, stride, {1, 1, 1, 1});

	auto res = model.add(branch1, branch2b);
	return model.relu(res);

}

int main()
{
    mv::Logger::setVerboseLevel(mv::Logger::VerboseLevel::VerboseDebug);

    // Define the primary compilation unit
    mv::CompilationUnit unit("ResNet18");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    // Compose the model for ResNet18
    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajorPlanar);
    auto conv1 = convBatchNormBlock(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3});
    conv1 = cm.relu(conv1);
    auto pool1 = cm.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    cm.output(pool1);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);
    
    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("cm_testblob.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string("testblob.blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = true;
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;
    
    // Initialize compilation 
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);

    mv::OpModel cm2(cm);
    // Run all passes
    unit.run();
    cm2.getBinaryBuffer()->dumpBuffer("final_RAM.blob") ;

    //system("dot -Tsvg cm_resnet18.dot -o cm_resnet18.svg");
    //system("dot -Tsvg cm_resnet18_adapt.dot -o cm_resnet18_adapt.svg");
    //system("dot -Tsvg cm_resnet18_final.dot -o cm_resnet18_final.svg");
    return 0;
}
