/**
 * @brief Example presenting composition of l8b network and its compilation.
 * Outputs a dot files that visualize the model before and during the compilation.
 * @file cm_l8b.cpp
 * @author Stanislaw Maciag
 * @date 2018-07-19
 */

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

/**
 * @brief Helper function creates a chain of conv2D, batchnorm and scale attached to the selected input tensor
 * 
 * @param model Master compositional model
 * @param input Tensor that is an input data for the conv2D
 * @param kernelShape Shape of conv2D kernel
 * @param stride Stride of conv2D
 * @param padding Padding of conv2D
 * @return mv::Data::TensorIterator Iterator referencing the created scale output 
 */
mv::Data::TensorIterator convBatchNormScaleBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input,  mv::Shape kernelShape,mv::UnsignedVector2D stride, mv::UnsignedVector4D padding)
{
    
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(kernelShape.totalSize());
    auto weights = model.constant(weightsData, kernelShape, mv::DType::Float, mv::Order::LastDimMajor);
    auto conv = model.conv2D(input, weights, stride, padding);

    // For debugging purpose weights are initialized as sequences of numbers, to be replaced with actual weights
    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
   
    auto bnmean = model.constant(meanData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    auto bnvariance = model.constant(varianceData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    auto bnoffset = model.constant(offsetData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    auto bnscale = model.constant(scaleData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    auto batchNorm = model.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);

    mv::dynamic_vector<mv::float_type> scaleParamData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    auto scaleParam = model.constant(scaleParamData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    return model.scale(batchNorm, scaleParam);

}


int main()
{

    // Define the primary compilation unit
    mv::CompilationUnit unit(mv::Logger::VerboseLevel::VerboseInfo);

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input(mv::Shape(224, 224, 3), mv::DType::Float, mv::Order::LastDimMajor);
    auto convBlock = convBatchNormScaleBlock(cm, input, mv::Shape(7, 7, 3, 64), {2, 2}, {3, 3, 3, 3});
    auto relu = cm.relu(convBlock);
    auto maxpool1 = cm.maxpool2D(relu, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto convBlock1 = convBatchNormScaleBlock(cm, maxpool1, mv::Shape(1, 1, 64, 64), {1, 1}, {0, 0, 0, 0});
    auto convBlock2a = convBatchNormScaleBlock(cm, maxpool1, mv::Shape(3, 3, 64, 64), {1, 1}, {1, 1, 1, 1});
    auto relu2a = cm.relu(convBlock2a);
    auto convBlock2b = convBatchNormScaleBlock(cm, relu2a, mv::Shape(3, 3, 64, 64), {1, 1}, {1, 1, 1, 1});
    auto add = cm.add(convBlock1, convBlock2b);
    auto maxpool2 = cm.maxpool2D(add, {1, 1}, {1, 1}, {0, 0, 0, 0});
    cm.output(maxpool2);

    // Load target descriptor for the selected target to the compilation unit
    std::string targetDescPath = std::getenv("MCM_HOME") + std::string("/config/target/ma2480.json");
    unit.targetDescriptor().load(targetDescPath);

    // Schedule pass for blob generation
    unit.targetDescriptor().appendSerialPass("GenerateBlob");
    
    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("cm_l8b.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;

    // Define output blob file name
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("cm_l8b.blob");
    
    // Initialize compilation 
    unit.initialize();

    // Run all passes
    unit.run();

    //system("dot -Tsvg cm_l8b.dot -o cm_l8b.svg");
    //system("dot -Tsvg cm_l8b_adapt.dot -o cm_l8b_adapt.svg");

    return 0;

}