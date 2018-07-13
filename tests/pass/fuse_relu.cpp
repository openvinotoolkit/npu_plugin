#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/transform/fuse_relu.hpp"

TEST(fuse_relu, case_conv)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::ColumnMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    auto relu = om.relu(conv);
    auto reluOp = om.getSourceOp(relu);
    om.output(relu);
    
    auto outputOp = reluOp.leftmostChild();

    mv::pass::FuseReLU fuseRelu;
    fuseRelu.run(om);

    // Check if relu components were invalidated
    /*ASSERT_FALSE(om.isValid(relu));
    ASSERT_FALSE(om.isValid(reluOp));*/

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 3);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    ASSERT_TRUE(convOp->hasAttr("postOpType"));
    ASSERT_EQ(convOp->getAttr("postOpType").getContent<mv::OpType>(), mv::OpType::ReLU);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(convOp);

    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}