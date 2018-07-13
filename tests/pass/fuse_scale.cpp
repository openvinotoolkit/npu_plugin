#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/transform/fuse_scale.hpp"
#include "include/mcm/deployer/fstd_ostream.hpp"
#include "include/mcm/pass/deploy/generate_dot.hpp"
TEST(fuse_scale, case_conv)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    mv::dynamic_vector<mv::float_type> scalesData = mv::utils::generateSequence<mv::float_type>(32);
    auto scales = om.constant(scalesData, mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, "biases");
    auto scale = om.scale(conv, scales);
    auto scaleOp = om.getSourceOp(scale);
    om.output(scale);
    
    auto outputOp = scaleOp.leftmostChild();
    mv::FStdOStream ostream("cm1.dot");
    mv::pass::GenerateDot generateDot(ostream, mv::pass::GenerateDot::OutputScope::ExecOpControlModel, mv::pass::GenerateDot::ContentLevel::ContentFull);
    bool dotResult = generateDot.run(om);    
    if (dotResult)
        (void)system("dot -Tsvg cm1.dot -o cm1.svg");
    mv::pass::FuseScale fuseScale;
    fuseScale.run(om);
    ostream.setFileName("cm2.dot");
    dotResult = generateDot.run(om);    
    if (dotResult)
        (void)system("dot -Tsvg cm2.dot -o cm2.svg");
    // Check if bias components were invalidated
    /*ASSERT_FALSE(om.isValid(biases));
    ASSERT_FALSE(om.isValid(bias));
    ASSERT_FALSE(om.isValid(biasOp));*/

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 3);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    mv::Tensor scaleParam("scale", {32}, mv::DType::Float, mv::Order::LastDimMajor, scalesData);
    mv::Tensor originalWeights("originalWeights", mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, weightsData);
    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);

    for (unsigned i = 0; i < convOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getInputTensor(1)->getData()[i], newWeigths.getData()[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(convOp);

    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}