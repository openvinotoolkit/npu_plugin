#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/deployer/fstd_ostream.hpp"
#include "include/mcm/pass/deploy/generate_dot.hpp"
#include "include/mcm/pass/transform/fuse_batch_norm.hpp"
#include "include/mcm/pass/transform/fuse_bias.hpp"
#include "include/mcm/pass/transform/fuse_relu.hpp"

mv::Data::TensorIterator convBatchNormBlock(mv::OpModel& model, mv::Data::TensorIterator input,  mv::Shape kernelShape, mv::UnsignedVector2D stride, mv::UnsignedVector4D padding)
{
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(kernelShape.totalSize());
    auto weights = model.constant(weightsData, kernelShape, mv::DType::Float, mv::Order::LastDimMajor);
    auto conv = model.conv2D(input, weights, stride, padding);

    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(conv->getShape()[-1]);
    auto bnmean = model.constant(meanData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    auto bnvariance = model.constant(varianceData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    auto bnoffset = model.constant(offsetData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    auto bnscale = model.constant(scaleData, conv->getShape()[-1], mv::DType::Float, mv::Order::LastDimMajor);
    return model.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
}

mv::Data::TensorIterator residualBlock(mv::OpModel& model, mv::Data::TensorIterator input)
{

    auto inputShape = input->getShape();
    auto branch2a = convBatchNormBlock(model, input, mv::Shape(3, 3, inputShape[2], inputShape[2]), {1, 1}, {1, 1, 1, 1});
    branch2a = model.relu(branch2a);
    auto branch2b = convBatchNormBlock(model, branch2a, mv::Shape(3, 3, inputShape[2], inputShape[2]), {1, 1}, {1, 1, 1, 1});

    auto res = model.add(input, branch2b);
    return model.relu(res);

}

mv::Data::TensorIterator residualConvBlock(mv::OpModel& model, mv::Data::TensorIterator input, mv::unsigned_type outputDepth, mv::UnsignedVector2D stride)
{

    auto inputShape = input->getShape();
    auto branch1 = convBatchNormBlock(model, input, mv::Shape(1, 1, inputShape[2], outputDepth), stride, {0, 0, 0, 0});
    auto branch2a = convBatchNormBlock(model, input, mv::Shape(3, 3, inputShape[2], outputDepth), {1, 1}, {1, 1, 1, 1});
    branch2a = model.relu(branch2a);
    auto branch2b = convBatchNormBlock(model, branch2a, mv::Shape(3, 3, outputDepth, outputDepth), stride, {1, 1, 1, 1});

    auto res = model.add(branch1, branch2b);
    return model.relu(res);

}

int main()
{
    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = om.input(mv::Shape(224, 224, 3), mv::DType::Float, mv::Order::LastDimMajor);

    auto conv1 = convBatchNormBlock(om, input, mv::Shape(7, 7, 3, 64), {2, 2}, {3, 3, 3, 3});
    conv1 = om.relu(conv1);
    auto pool1 = om.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto res2a = residualConvBlock(om, pool1, 64, {1, 1});
    auto res2b = residualBlock(om, res2a);
    auto res3a = residualConvBlock(om, res2b, 128, {2, 2});
    auto res3b = residualBlock(om, res3a);
    auto res4a = residualConvBlock(om, res3b, 256, {2, 2});
    auto res4b = residualBlock(om, res4a);
    auto res5a = residualConvBlock(om, res4b, 512, {2, 2});
    auto res5b = residualBlock(om, res5a);
    auto pool5 = om.avgpool2D(res5b, {7, 7}, {1, 1,}, {0, 0, 0, 0});
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(pool5->getShape().totalSize() * 1000u);
    auto weights = om.constant(weightsData, mv::Shape(pool5->getShape().totalSize(), 1000), mv::DType::Float, mv::Order::LastDimMajor);
    auto fc1000 = om.fullyConnected(pool5, weights);
    auto softmax = om.softmax(fc1000);

    om.output(softmax);

    mv::FStdOStream ostream("cm1.dot");
    mv::pass::GenerateDot generateDot(ostream, mv::pass::GenerateDot::OutputScope::ExecOpControlModel, mv::pass::GenerateDot::ContentLevel::ContentFull);
    bool dotResult = generateDot.run(om);    
    if (dotResult)
        (void)system("dot -Tsvg cm1.dot -o cm1.svg");

    mv::pass::FuseBatchNorm fuseBatchNorm;
    fuseBatchNorm.run(om);

    mv::pass::FuseBias fuseBias;
    fuseBias.run(om);

    mv::pass::FuseReLU fuseReLU;
    fuseReLU.run(om);
    
    ostream.setFileName("cm2.dot");
    dotResult = generateDot.run(om);    
    if (dotResult)
        (void)system("dot -Tsvg cm2.dot -o cm2.svg");

    return 0;

}