#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/transform/fuse_batch_norm.hpp"

TEST(fuse_batch_norm_pass, case_ndim_conv)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 3 * 3);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 3, 3), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    auto convShape = conv->getShape();
    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(convShape.totalSize());
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(convShape.totalSize());
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(convShape.totalSize());
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(convShape.totalSize());
    float eps = 1e-3;
    auto bnmean = om.constant(meanData, convShape, mv::DType::Float, mv::Order::LastDimMajor, "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, convShape, mv::DType::Float, mv::Order::LastDimMajor, "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, convShape, mv::DType::Float, mv::Order::LastDimMajor, "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, convShape, mv::DType::Float, mv::Order::LastDimMajor, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::pass::FuseBatchNorm fuseBatchNorm;
    fuseBatchNorm.run(om);

    // Check if batchnorm components were invalidated
    ASSERT_FALSE(om.isValid(batchnormOp));
    ASSERT_FALSE(om.isValid(batchnorm));
    ASSERT_FALSE(om.isValid(bnmeanOp));
    ASSERT_FALSE(om.isValid(bnmean));
    ASSERT_FALSE(om.isValid(bnvarianceOp));
    ASSERT_FALSE(om.isValid(bnvariance));
    ASSERT_FALSE(om.isValid(bnoffsetOp));
    ASSERT_FALSE(om.isValid(bnoffset));
    ASSERT_FALSE(om.isValid(bnscaleOp));
    ASSERT_FALSE(om.isValid(bnscale));

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
    mv::Tensor mean("mean", convShape, mv::DType::Float, mv::Order::LastDimMajor, meanData);
    mv::Tensor variance("variance", convShape, mv::DType::Float, mv::Order::LastDimMajor, varianceData);
    mv::Tensor offset("offset", convShape, mv::DType::Float, mv::Order::LastDimMajor, offsetData);
    mv::Tensor scale("scale", convShape, mv::DType::Float, mv::Order::LastDimMajor, scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_EQ(mulOp->getInputTensor(1)->getData().size(), scaleParam.getData().size());
    ASSERT_EQ(addOp->getInputTensor(1)->getData().size(), offsetParam.getData().size());

    for (unsigned i = 0; i < mulOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(mulOp->getInputTensor(1)->getData()[i], scaleParam.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(convOp);

    ++cIt;
    ASSERT_EQ(*mulOp, *cIt);
    ++cIt;
    ASSERT_EQ(*addOp, *cIt);
    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}

TEST(fuse_batch_norm_pass, case_1dim_conv)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    auto convShape = conv->getShape();
    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(convShape[-1]);
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(convShape[-1]);
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(convShape[-1]);
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(convShape[-1]);
    float eps = 1e-3;
    auto bnmean = om.constant(meanData, convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::pass::FuseBatchNorm fuseBatchNorm;
    fuseBatchNorm.run(om);

    // Check if batchnorm components were invalidated
    ASSERT_FALSE(om.isValid(batchnormOp));
    ASSERT_FALSE(om.isValid(batchnorm));
    ASSERT_FALSE(om.isValid(bnmeanOp));
    ASSERT_FALSE(om.isValid(bnmean));
    ASSERT_FALSE(om.isValid(bnvarianceOp));
    ASSERT_FALSE(om.isValid(bnvariance));
    ASSERT_FALSE(om.isValid(bnoffsetOp));
    ASSERT_FALSE(om.isValid(bnoffset));
    ASSERT_FALSE(om.isValid(bnscaleOp));
    ASSERT_FALSE(om.isValid(bnscale));

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
    mv::Tensor mean("mean", convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, meanData);
    mv::Tensor variance("variance", convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, varianceData);
    mv::Tensor offset("offset", convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, offsetData);
    mv::Tensor scale("scale", convShape[-1], mv::DType::Float, mv::Order::LastDimMajor, scaleData);
    mv::Tensor originalWeights("originalWeights", mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, weightsData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);

    for (unsigned i = 0; i < convOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getInputTensor(1)->getData()[i], newWeigths.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(convOp);

    ++cIt;
    ASSERT_EQ(*addOp, *cIt);
    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}

TEST(fuse_batch_norm_pass, case_ndim_nonconv)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::LastDimMajor);
    auto pool = om.maxpool2D(input, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto poolOp = om.getSourceOp(pool);
    auto poolShape = pool->getShape();
    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(poolShape.totalSize());
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(poolShape.totalSize());
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(poolShape.totalSize());
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(poolShape.totalSize());
    float eps = 1e-3;
    auto bnmean = om.constant(meanData, poolShape, mv::DType::Float, mv::Order::LastDimMajor, "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, poolShape, mv::DType::Float, mv::Order::LastDimMajor, "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, poolShape, mv::DType::Float, mv::Order::LastDimMajor, "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, poolShape, mv::DType::Float, mv::Order::LastDimMajor, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(pool, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::pass::FuseBatchNorm fuseBatchNorm;
    fuseBatchNorm.run(om);

    // Check if batchnorm components were invalidated
    ASSERT_FALSE(om.isValid(batchnormOp));
    ASSERT_FALSE(om.isValid(batchnorm));
    ASSERT_FALSE(om.isValid(bnmeanOp));
    ASSERT_FALSE(om.isValid(bnmean));
    ASSERT_FALSE(om.isValid(bnvarianceOp));
    ASSERT_FALSE(om.isValid(bnvariance));
    ASSERT_FALSE(om.isValid(bnoffsetOp));
    ASSERT_FALSE(om.isValid(bnoffset));
    ASSERT_FALSE(om.isValid(bnscaleOp));
    ASSERT_FALSE(om.isValid(bnscale));

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
    mv::Tensor mean("mean", poolShape, mv::DType::Float, mv::Order::LastDimMajor, meanData);
    mv::Tensor variance("variance", poolShape, mv::DType::Float, mv::Order::LastDimMajor, varianceData);
    mv::Tensor offset("offset", poolShape, mv::DType::Float, mv::Order::LastDimMajor, offsetData);
    mv::Tensor scale("scale", poolShape, mv::DType::Float, mv::Order::LastDimMajor, scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_EQ(mulOp->getInputTensor(1)->getData().size(), scaleParam.getData().size());
    ASSERT_EQ(addOp->getInputTensor(1)->getData().size(), offsetParam.getData().size());

    for (unsigned i = 0; i < mulOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(mulOp->getInputTensor(1)->getData()[i], scaleParam.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(poolOp);

    ++cIt;
    ASSERT_EQ(*mulOp, *cIt);
    ++cIt;
    ASSERT_EQ(*addOp, *cIt);
    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}

TEST(fuse_batch_norm_pass, case_1dim_nonconv)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::LastDimMajor);
    auto pool = om.maxpool2D(input, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto poolOp = om.getSourceOp(pool);
    auto poolShape = pool->getShape();
    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(poolShape[-1]);
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(poolShape[-1]);
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(poolShape[-1]);
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(poolShape[-1]);
    float eps = 1e-3;
    auto bnmean = om.constant(meanData, poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNorm(pool, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::pass::FuseBatchNorm fuseBatchNorm;
    fuseBatchNorm.run(om);

    // Check if batchnorm components were invalidated
    ASSERT_FALSE(om.isValid(batchnormOp));
    ASSERT_FALSE(om.isValid(batchnorm));
    ASSERT_FALSE(om.isValid(bnmeanOp));
    ASSERT_FALSE(om.isValid(bnmean));
    ASSERT_FALSE(om.isValid(bnvarianceOp));
    ASSERT_FALSE(om.isValid(bnvariance));
    ASSERT_FALSE(om.isValid(bnoffsetOp));
    ASSERT_FALSE(om.isValid(bnoffset));
    ASSERT_FALSE(om.isValid(bnscaleOp));
    ASSERT_FALSE(om.isValid(bnscale));

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
    mv::Tensor mean("mean", poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, meanData);
    mv::Tensor variance("variance", poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, varianceData);
    mv::Tensor offset("offset", poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, offsetData);
    mv::Tensor scale("scale", poolShape[-1], mv::DType::Float, mv::Order::LastDimMajor, scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_EQ(mulOp->getInputTensor(1)->getData().size(), scaleParam.getData().size());
    ASSERT_EQ(addOp->getInputTensor(1)->getData().size(), offsetParam.getData().size());

    for (unsigned i = 0; i < mulOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(mulOp->getInputTensor(1)->getData()[i], scaleParam.getData()[i]);

    for (unsigned i = 0; i < addOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(addOp->getInputTensor(1)->getData()[i], offsetParam.getData()[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(poolOp);

    ++cIt;
    ASSERT_EQ(*mulOp, *cIt);
    ++cIt;
    ASSERT_EQ(*addOp, *cIt);
    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}