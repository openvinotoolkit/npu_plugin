#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

class layers_reorg_yolo:
    public LayersTest<std::tuple<unsigned, mv::Shape, mv::DType, mv::Target>> {};

TEST_P(layers_reorg_yolo, dump_blob)
{
    auto param = GetParam();
    auto stride   = std::get<0>(param);
    auto in_shape = std::get<1>(param);
    auto dtype    = std::get<2>(param);
    auto target   = std::get<3>(param);

    std::string test_name = "layers_reorg_yolo_" + testToString(in_shape)
                          + "_" + dtype.toString() + "_" + testToString(target);
    std::cout << "Test: " << test_name << std::endl;

    testSetName(test_name);

    mv::Order in_order("NCHW");
    mv::Order out_order("NC");

    assert(in_shape.ndims() == 4);
    auto W = in_shape[0];
    auto H = in_shape[1];
    auto C = in_shape[2];
    auto N = in_shape[3];

    assert(stride > 0);
    assert((W % stride == 0) && (H % stride == 0));
    mv::Shape out_shape({W/stride, H/stride, C*stride*stride, N});

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto  input = om.input(in_shape, dtype, in_order);
    auto  reorg = om.reorgYolo(input, stride);
    auto output = om.output(reorg);

    ASSERT_TRUE(om.isValid(reorg));
    ASSERT_TRUE(om.isValid(om.getSourceOp(reorg)));

    auto compDesc = testGetCompilationDescriptor(unit, target);

    EXPECT_EQ("OK", testSetGenBlob(compDesc));
    EXPECT_EQ("OK", testSetGenDot(compDesc));

    ASSERT_TRUE(unit.loadTargetDescriptor(target));
    ASSERT_TRUE(unit.initialize());

    // C++ exception if fails
    auto result = unit.run();

    EXPECT_EQ("OK", testDumpJson(result));
    EXPECT_EQ("OK", testDumpBlob());
    EXPECT_EQ("OK", testDumpDot());
}

using namespace testing;

INSTANTIATE_TEST_CASE_P(demo, layers_reorg_yolo,
                        Combine(Values(2),
                                Values(mv::Shape({320, 200,  3, 8}),
                                       mv::Shape({ 26,  26, 24, 8})),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
