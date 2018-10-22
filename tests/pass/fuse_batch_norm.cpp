#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(fuse_batch_norm_pass, case_ndim_conv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 3}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 3 * 3);
    auto weights = om.constant(weightsData, {3, 3, 3, 3}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    auto convShape = conv->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(convShape.totalSize());
    std::vector<double> varianceData = mv::utils::generateSequence<double>(convShape.totalSize());
    std::vector<double> offsetData = mv::utils::generateSequence<double>(convShape.totalSize());
    std::vector<double> scaleData = mv::utils::generateSequence<double>(convShape.totalSize());
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyCompDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 8);
    ASSERT_EQ(dm.tensorsCount(), 7);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    // Check replacament for batchnorm multiplicative component
    auto mulOp = convOp.leftmostChild();
    ASSERT_EQ(mulOp->getOpType(), mv::OpType::Multiply);
    ASSERT_EQ(mulOp.childrenSize(), 1);
    ASSERT_TRUE(mulOp->getInputTensor(1)->isPopulated());

    // Check replacement for batchnorm additive component
    auto addOp = mulOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), mv::OpType::Add);
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), meanData);
    mv::Tensor variance("variance", convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), varianceData);
    mv::Tensor offset("offset", convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), offsetData);
    mv::Tensor scale("scale", convShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_EQ(mulOp->getInputTensor(1)->getData().size(), scaleParam.getData().size());
    ASSERT_EQ(addOp->getInputTensor(1)->getData().size(), offsetParam.getData().size());

    for (unsigned i = 0; i < mulOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(mulOp->getInputTensor(1)->getData()[i], scaleParam.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);
   
}

TEST(fuse_batch_norm_pass, case_1dim_conv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 16}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, {3, 3, 16, 32}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    auto convShape = conv->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(convShape[-1]);
    std::vector<double> varianceData = mv::utils::generateSequence<double>(convShape[-1]);
    std::vector<double> offsetData = mv::utils::generateSequence<double>(convShape[-1]);
    std::vector<double> scaleData = mv::utils::generateSequence<double>(convShape[-1]);
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyCompDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 6);
    ASSERT_EQ(dm.tensorsCount(), 5);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);

    // Check replacement for batchnorm additive component
    auto addOp = convOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), mv::OpType::Bias);
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), meanData);
    mv::Tensor variance("variance", {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), varianceData);
    mv::Tensor offset("offset", {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), offsetData);
    mv::Tensor scale("scale", {convShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), scaleData);
    mv::Tensor originalWeights("originalWeights", {3, 3, 16, 32}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), weightsData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);

    for (unsigned i = 0; i < convOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getInputTensor(1)->getData()[i], newWeigths.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);

}

TEST(fuse_batch_norm_pass, case_ndim_nonconv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 3}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    auto pool = om.maxpool2D(input, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto poolOp = om.getSourceOp(pool);
    auto poolShape = pool->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(poolShape.totalSize());
    std::vector<double> varianceData = mv::utils::generateSequence<double>(poolShape.totalSize());
    std::vector<double> offsetData = mv::utils::generateSequence<double>(poolShape.totalSize());
    std::vector<double> scaleData = mv::utils::generateSequence<double>(poolShape.totalSize());
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(pool, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyCompDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 7);
    ASSERT_EQ(dm.tensorsCount(), 6);

    // Check predecessing operation
    ASSERT_EQ(poolOp.childrenSize(), 1);
    
    // Check replacament for batchnorm multiplicative component
    auto mulOp = poolOp.leftmostChild();
    ASSERT_EQ(mulOp->getOpType(), mv::OpType::Multiply);
    ASSERT_EQ(mulOp.childrenSize(), 1);
    ASSERT_TRUE(mulOp->getInputTensor(1)->isPopulated());

    // Check replacement for batchnorm additive component
    auto addOp = mulOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), mv::OpType::Add);
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), meanData);
    mv::Tensor variance("variance", poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), varianceData);
    mv::Tensor offset("offset", poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), offsetData);
    mv::Tensor scale("scale", poolShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_EQ(mulOp->getInputTensor(1)->getData().size(), scaleParam.getData().size());
    ASSERT_EQ(addOp->getInputTensor(1)->getData().size(), offsetParam.getData().size());

    for (unsigned i = 0; i < mulOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(mulOp->getInputTensor(1)->getData()[i], scaleParam.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);

}

TEST(fuse_batch_norm_pass, case_1dim_nonconv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 16}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    auto pool = om.maxpool2D(input, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto poolOp = om.getSourceOp(pool);
    auto poolShape = pool->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(poolShape[-1]);
    std::vector<double> varianceData = mv::utils::generateSequence<double>(poolShape[-1]);
    std::vector<double> offsetData = mv::utils::generateSequence<double>(poolShape[-1]);
    std::vector<double> scaleData = mv::utils::generateSequence<double>(poolShape[-1]);
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(pool, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyCompDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 7);
    ASSERT_EQ(dm.tensorsCount(), 6);

    // Check predecessing operation
    ASSERT_EQ(poolOp.childrenSize(), 1);
    
    // Check replacament for batchnorm multiplicative component
    auto mulOp = poolOp.leftmostChild();
    ASSERT_EQ(mulOp->getOpType(), mv::OpType::Scale);
    ASSERT_EQ(mulOp.childrenSize(), 1);
    ASSERT_TRUE(mulOp->getInputTensor(1)->isPopulated());

    // Check replacement for batchnorm additive component
    auto addOp = mulOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), mv::OpType::Bias);
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), meanData);
    mv::Tensor variance("variance", {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), varianceData);
    mv::Tensor offset("offset", {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), offsetData);
    mv::Tensor scale("scale", {poolShape[-1]}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_EQ(mulOp->getInputTensor(1)->getData().size(), scaleParam.getData().size());
    ASSERT_EQ(addOp->getInputTensor(1)->getData().size(), offsetParam.getData().size());

    for (unsigned i = 0; i < mulOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(mulOp->getInputTensor(1)->getData()[i], scaleParam.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);

}
