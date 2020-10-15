#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(fuse_batch_norm_pass, case_ndim_conv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 3 * 3);

    auto weights = om.constant(weightsData, {3, 3, 3, 3}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)), {{},{},{},{}}, "weights");
    auto conv = om.conv(input, weights, {1, 1}, {1, 1, 1, 1}, 1);
    auto convOp = om.getSourceOp(conv);
    auto convShape = conv->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(convShape.totalSize());
    std::vector<double> varianceData = mv::utils::generateSequence<double>(convShape.totalSize());
    std::vector<double> offsetData = mv::utils::generateSequence<double>(convShape.totalSize());
    std::vector<double> scaleData = mv::utils::generateSequence<double>(convShape.totalSize());
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, convShape, mv::DType("Float16"), conv->getOrder() ,{{},{},{},{}}, "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, convShape, mv::DType("Float16"), conv->getOrder() , {{},{},{},{}},"variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, convShape, mv::DType("Float16"), conv->getOrder() ,{{},{},{},{}}, "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, convShape, mv::DType("Float16"), conv->getOrder(), {{},{},{},{}}, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNormalization(conv, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 8);
    ASSERT_EQ(dm.tensorsCount(), 7);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    // Check replacament for batchnorm multiplicative component
    auto mulOp = convOp.leftmostChild();
    ASSERT_EQ(mulOp->getOpType(), "Multiply");
    ASSERT_EQ(mulOp.childrenSize(), 1);
    ASSERT_TRUE(mulOp->getInputTensor(1)->isPopulated());

    // Check replacement for batchnorm additive component
    auto addOp = mulOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), "Add");
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", convShape, mv::DType("Float16"), mv::Order("NCHW"), meanData);
    mv::Tensor variance("variance", convShape, mv::DType("Float16"), mv::Order("NCHW"), varianceData);
    mv::Tensor offset("offset", convShape, mv::DType("Float16"), mv::Order("NCHW"), offsetData);
    mv::Tensor scale("scale", convShape, mv::DType("Float16"), mv::Order("NCHW"), scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset,
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    auto mulOpData = mulOp->getInputTensor(1)->getDoubleData();
    auto mulOpDataSize = mulOpData.size();

    auto scaleParamData = scaleParam.getDoubleData();
    auto scaleParamDataSize = scaleParamData.size();

    auto addOpData = addOp->getInputTensor(1)->getDoubleData();
    auto addOpDataSize = addOpData.size();

    auto newOffsetData = offsetParam.getDoubleData();
    auto newOffsetDataSize = newOffsetData.size();

    ASSERT_EQ(mulOpDataSize, scaleParamDataSize);
    ASSERT_EQ(addOpDataSize, newOffsetDataSize);

    for (unsigned i = 0; i < mulOpDataSize; ++i)
        ASSERT_FLOAT_EQ(mulOpData[i], scaleParamData[i]);

    for (unsigned i = 0; i < addOpDataSize; ++i)
        ASSERT_FLOAT_EQ(addOpData[i], newOffsetData[i]);
   
}

TEST(fuse_batch_norm_pass, case_1dim_conv)
{
    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);

    auto weights = om.constant(weightsData, {3, 3, 16, 32}, mv::DType("Float16"), mv::Order("NCHW"), {{},{},{},{}},"weights");
    auto conv = om.conv(input, weights, {1, 1}, {1, 1, 1, 1}, 1);
    auto convOp = om.getSourceOp(conv);
    auto convShape = conv->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(convShape[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> varianceData = mv::utils::generateSequence<double>(convShape[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> offsetData = mv::utils::generateSequence<double>(convShape[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> scaleData = mv::utils::generateSequence<double>(convShape[mv::IO_CHANNEL_DIMENSION]);
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), {{},{},{},{}},"mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNormalization(conv, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 6);
    ASSERT_EQ(dm.tensorsCount(), 5);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);

    // Check replacement for batchnorm additive component
    auto addOp = convOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), "Bias");
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), meanData);
    mv::Tensor variance("variance", {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), varianceData);
    mv::Tensor offset("offset", {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), offsetData);
    mv::Tensor scale("scale", {convShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), scaleData);
    mv::Tensor originalWeights("originalWeights", {3, 3, 16, 32}, mv::DType("Float16"), mv::Order("NCHW"), weightsData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset, 
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);

    auto newWeigthsData = newWeigths.getDoubleData();
    auto originalWeightsFromConv = convOp->getInputTensor(1);
    auto originalWeightsFromConvData = originalWeightsFromConv->getDoubleData();

    auto addOpData = addOp->getInputTensor(1)->getDoubleData();
    auto offsetParamData = offsetParam.getDoubleData();

    for (unsigned i = 0; i < originalWeightsFromConvData.size(); ++i)
        ASSERT_FLOAT_EQ(originalWeightsFromConvData[i], newWeigthsData[i]);
        
    for (unsigned i = 0; i < addOpData.size(); ++i)
        ASSERT_FLOAT_EQ(addOpData[i], offsetParamData[i]);

}

TEST(fuse_batch_norm_pass, case_ndim_nonconv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto pool = om.maxPool(input, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto poolOp = om.getSourceOp(pool);
    auto poolShape = pool->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(poolShape.totalSize());
    std::vector<double> varianceData = mv::utils::generateSequence<double>(poolShape.totalSize());
    std::vector<double> offsetData = mv::utils::generateSequence<double>(poolShape.totalSize());
    std::vector<double> scaleData = mv::utils::generateSequence<double>(poolShape.totalSize());
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, poolShape, mv::DType("Float16"), mv::Order("NCHW"),{{},{},{},{}}, "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, poolShape, mv::DType("Float16"), mv::Order("NCHW"),{{},{},{},{}}, "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, poolShape, mv::DType("Float16"), mv::Order("NCHW"), {{},{},{},{}},"offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, poolShape, mv::DType("Float16"), mv::Order("NCHW"),{{},{},{},{}}, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNormalization(pool, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 7);
    ASSERT_EQ(dm.tensorsCount(), 6);

    // Check predecessing operation
    ASSERT_EQ(poolOp.childrenSize(), 1);
    
    // Check replacament for batchnorm multiplicative component
    auto mulOp = poolOp.leftmostChild();
    ASSERT_EQ(mulOp->getOpType(), "Multiply");
    ASSERT_EQ(mulOp.childrenSize(), 1);
    ASSERT_TRUE(mulOp->getInputTensor(1)->isPopulated());

    // Check replacement for batchnorm additive component
    auto addOp = mulOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), "Add");
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", poolShape, mv::DType("Float16"), mv::Order("NCHW"), meanData);
    mv::Tensor variance("variance", poolShape, mv::DType("Float16"), mv::Order("NCHW"), varianceData);
    mv::Tensor offset("offset", poolShape, mv::DType("Float16"), mv::Order("NCHW"), offsetData);
    mv::Tensor scale("scale", poolShape, mv::DType("Float16"), mv::Order("NCHW"), scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset,
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_TRUE(mulOp->getInputTensor(1)->isDoubleType());
    auto mulOpData = mulOp->getInputTensor(1)->getDoubleData();
    auto mulOpDataSize = mulOpData.size();

    ASSERT_TRUE(addOp->getInputTensor(1)->isDoubleType());
    auto addOpData = addOp->getInputTensor(1)->getDoubleData();
    auto addOpDataSize = addOpData.size();

    ASSERT_TRUE(offsetParam.isDoubleType());
    auto offsetParamData = offsetParam.getDoubleData();
    auto offsetParamDataSize = offsetParamData.size();

    ASSERT_TRUE(scaleParam.isDoubleType());
    auto scaleParamData = scaleParam.getDoubleData();
    auto scaleParamDataSize = scaleParamData.size();

    ASSERT_EQ(mulOpDataSize, scaleParamDataSize);
    ASSERT_EQ(addOpDataSize, offsetParamDataSize);

    for (unsigned i = 0; i < mulOpDataSize; ++i)
        ASSERT_FLOAT_EQ(mulOpData[i], scaleParamData[i]);

    for (unsigned i = 0; i < addOpDataSize; ++i)
        ASSERT_FLOAT_EQ(addOpData[i], offsetParamData[i]);

}

TEST(fuse_batch_norm_pass, case_1dim_nonconv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto pool = om.maxPool(input, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto poolOp = om.getSourceOp(pool);
    auto poolShape = pool->getShape();
    std::vector<double> meanData = mv::utils::generateSequence<double>(poolShape[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> varianceData = mv::utils::generateSequence<double>(poolShape[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> offsetData = mv::utils::generateSequence<double>(poolShape[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> scaleData = mv::utils::generateSequence<double>(poolShape[mv::IO_CHANNEL_DIMENSION]);
    double eps = 1e-3;
    auto bnmean = om.constant(meanData, {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "mean");
    auto bnmeanOp = om.getSourceOp(bnmean);
    auto bnvariance = om.constant(varianceData, {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "variance");
    auto bnvarianceOp = om.getSourceOp(bnvariance);
    auto bnoffset = om.constant(offsetData, {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), {{},{},{},{}},"offset");
    auto bnoffsetOp = om.getSourceOp(bnoffset);
    auto bnscale = om.constant(scaleData, {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "scale");
    auto bnscaleOp = om.getSourceOp(bnscale);
    auto batchnorm = om.batchNormalization(pool, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto batchnormOp = om.getSourceOp(batchnorm);

    om.output(batchnorm);
    auto outputOp = batchnormOp.leftmostChild();

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseBatchNorm")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 7);
    ASSERT_EQ(dm.tensorsCount(), 6);

    // Check predecessing operation
    ASSERT_EQ(poolOp.childrenSize(), 1);
    
    // Check replacament for batchnorm multiplicative component
    auto mulOp = poolOp.leftmostChild();
    ASSERT_EQ(mulOp->getOpType(), "Scale");
    ASSERT_EQ(mulOp.childrenSize(), 1);
    ASSERT_TRUE(mulOp->getInputTensor(1)->isPopulated());

    // Check replacement for batchnorm additive component
    auto addOp = mulOp.leftmostChild();
    ASSERT_EQ(addOp->getOpType(), "Bias");
    ASSERT_EQ(addOp.childrenSize(), 1);
    ASSERT_TRUE(addOp->getInputTensor(1)->isPopulated());

    // Check fusing
    mv::Tensor mean("mean", {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), meanData);
    mv::Tensor variance("variance", {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), varianceData);
    mv::Tensor offset("offset", {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), offsetData);
    mv::Tensor scale("scale", {poolShape[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"), scaleData);

    mv::Tensor scaleParam = mv::math::divide(scale, mv::math::sqrt(mv::math::add(variance, eps)));
    mv::Tensor offsetParam = mv::math::subtract(offset,
        mv::math::divide(mv::math::multiply(scale, mean), mv::math::sqrt(mv::math::add(variance, eps))));

    ASSERT_TRUE(mulOp->getInputTensor(1)->isDoubleType());

    auto mulOpData = mulOp->getInputTensor(1)->getDoubleData();
    auto mulOpDataSize = mulOpData.size();

    auto scaleParamData = scaleParam.getDoubleData();
    auto scaleParamDataSize = scaleParamData.size();

    auto addOpData = addOp->getInputTensor(1)->getDoubleData();
    auto addOpDataSize = addOpData.size();

    auto newOffsetData = offsetParam.getDoubleData();
    auto newOffsetDataSize = offsetData.size();

    ASSERT_EQ(mulOpDataSize, scaleParamDataSize);
    ASSERT_EQ(addOpDataSize, newOffsetDataSize);

    for (unsigned i = 0; i < mulOpDataSize; ++i)
        ASSERT_FLOAT_EQ(mulOpData[i], scaleParamData[i]);

    for (unsigned i = 0; i < addOpDataSize; ++i)
        ASSERT_FLOAT_EQ(addOpData[i], newOffsetData[i]);

}
