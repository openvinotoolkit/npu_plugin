#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/data_model.hpp"
#include "include/fathom/computation/model/control_model.hpp"
#include "include/fathom/computation/utils/data_generator.hpp"
#include "include/fathom/deployer/fstd_ostream.hpp"
#include "include/fathom/pass/deploy/dot_pass.hpp"


mv::DataContext::TensorIterator convBatchNormBlock(mv::OpModel &model, mv::DataContext::TensorIterator input,  mv::Shape kernelShape, mv::UnsignedVector2D stride, mv::UnsignedVector4D padding)
{
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(kernelShape.totalSize());
    auto weights = model.constant(weightsData, kernelShape, mv::DType::Float, mv::Order::NWHC);
    auto conv = model.conv2D(input, weights, stride, padding);

    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    auto bnmean = model.constant(meanData, conv->getShape(), mv::DType::Float, mv::Order::NWHC);
    auto bnvariance = model.constant(varianceData, conv->getShape(), mv::DType::Float, mv::Order::NWHC);
    auto bnoffset = model.constant(offsetData, conv->getShape(), mv::DType::Float, mv::Order::NWHC);
    auto bnscale = model.constant(scaleData, conv->getShape(), mv::DType::Float, mv::Order::NWHC);
    return model.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
}


int main()
{
    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = om.input(mv::Shape(224, 224, 3), mv::DType::Float, mv::Order::NWHC);

    auto conv1 = convBatchNormBlock(om, input, mv::Shape(7, 7, 3, 64), {2, 2}, {3, 3, 3, 3});
    conv1 = om.relu(conv1);
    auto pool1 = om.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});

    auto res2a_branch1 = convBatchNormBlock(om, pool1, mv::Shape(1, 1, 64, 64), {1, 1}, {0, 0, 0, 0});
    auto res2a_branch2a = convBatchNormBlock(om, pool1, mv::Shape(3, 3, 64, 64), {1, 1}, {1, 1, 1, 1});
    res2a_branch2a = om.relu(res2a_branch2a);
    auto res2a_branch2b = convBatchNormBlock(om, res2a_branch2a, mv::Shape(3, 3, 64, 64), {1, 1}, {1, 1, 1, 1});

    auto res2a = om.add(res2a_branch1, res2a_branch2b);
    res2a = om.relu(res2a);

    om.output(res2a);

    mv::FStdOStream ostream("cm.dot");
    mv::pass::DotPass dotPass(om.logger(), ostream, mv::pass::DotPass::OutputScope::OpControlModel, mv::pass::DotPass::ContentLevel::ContentFull);
    bool dotResult = dotPass.run(om);    
    if (dotResult)
        system("dot -Tsvg cm.dot -o cm.svg");

    return 0;

    /*mv::dynamic_vector<mv::float_type> weights2Data = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::dynamic_vector<mv::float_type> weights3Data = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);

    auto weights1 = om.constant(weights1Data, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC);
    auto conv1 = om.conv2D(input, weights1, {2, 2}, {1, 1, 1, 1});
    auto pool1 = om.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = om.constant(weights2Data, mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::NWHC);
    auto conv2 = om.conv2D(pool1, weights2, {2, 2}, {2, 2, 2, 2});
    auto pool2 = om.maxpool2D(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = om.constant(weights3Data, mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::NWHC);
    auto conv3 = om.conv2D(pool2, weights3, {1, 1}, {0, 0, 0, 0});
    auto output = om.output(conv3);*/

}