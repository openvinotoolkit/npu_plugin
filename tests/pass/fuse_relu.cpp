#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(fuse_relu, case_conv)
{

    mv::OpModel om("testModel");
    auto input = om.input({64, 64, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, {3, 3, 16, 32}, mv::DType("Float16"), mv::Order("NCHW"),{{},{},{},{}}, "weights");
    auto conv = om.conv(input, weights, {1, 1}, {1, 1, 1, 1}, 1);
    auto convOp = om.getSourceOp(conv);
    auto relu = om.relu(conv);
    auto reluOp = om.getSourceOp(relu);
    om.output(relu);
    
    auto outputOp = reluOp.leftmostChild();

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseRelu")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 3);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    ASSERT_TRUE(convOp->hasAttr("postOpType"));
    //ASSERT_EQ(convOp->get<mv::OpType>("postOpType"), mv::OpType::ReLU);

}
