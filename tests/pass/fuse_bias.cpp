#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/transform/fuse_bias.hpp"

TEST(fuse_bias, case_conv)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    mv::dynamic_vector<mv::float_type> biasesData = mv::utils::generateSequence<mv::float_type>(32);
    auto biases = om.constant(biasesData, mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, "biases");
    auto bias = om.bias(conv, biases);
    auto biasOp = om.getSourceOp(bias);
    om.output(bias);
    
    auto outputOp = biasOp.leftmostChild();

    mv::pass::FuseBias fuseBias;
    fuseBias.run(om);

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

    for (unsigned i = 0; i < convOp->getAttr("bias").getContent<mv::dynamic_vector<float>>().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getAttr("bias").getContent<mv::dynamic_vector<float>>()[i], biasesData[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(convOp);

    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}