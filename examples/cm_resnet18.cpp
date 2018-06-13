#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/data_model.hpp"
#include "include/fathom/computation/model/control_model.hpp"
#include "include/fathom/computation/utils/data_generator.hpp"
#include "include/fathom/deployer/fstd_ostream.hpp"
#include "include/fathom/pass/deploy/dot_pass.hpp"



int main()
{
    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = om.input(mv::Shape(224, 224, 3), mv::DType::Float, mv::Order::NWHC);

    mv::dynamic_vector<mv::float_type> weights1Data = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    auto weights1 = om.constant(weights1Data, mv::Shape(7, 7, 3, 64), mv::DType::Float, mv::Order::NWHC);
    auto conv1 = om.conv2D(input, weights1, {2, 2}, {3, 3, 3, 3});

    mv::dynamic_vector<mv::float_type> mean1Data = mv::utils::generateSequence<mv::float_type>(conv1->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> variance1Data = mv::utils::generateSequence<mv::float_type>(conv1->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> offset1Data = mv::utils::generateSequence<mv::float_type>(conv1->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> scale1Data = mv::utils::generateSequence<mv::float_type>(conv1->getShape().totalSize());
    auto bnmean1 = om.constant(mean1Data, conv1->getShape(), mv::DType::Float, mv::Order::NWHC);
    auto bnvariance1 = om.constant(variance1Data, conv1->getShape(), mv::DType::Float, mv::Order::NWHC);
    auto bnoffset1 = om.constant(offset1Data, conv1->getShape(), mv::DType::Float, mv::Order::NWHC);
    auto bnscale1 = om.constant(scale1Data, conv1->getShape(), mv::DType::Float, mv::Order::NWHC);
    auto batchNorm1 = om.batchNorm(conv1, bnmean1, bnvariance1, bnoffset1, bnscale1, 1e-6);
    auto relu1 = om.relu(batchNorm1);

    om.output(relu1);

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