#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

using Minmax = typename std::array<double, 2>;

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

class layers_clamp:
    public LayersTest<std::tuple<Minmax, Form, mv::DType, mv::Target>> {};

TEST_P(layers_clamp, dump_blob)
{
    auto param = GetParam();
    auto minmax = std::get<0>(param);
    auto form   = std::get<1>(param);
    auto dtype  = std::get<2>(param);
    auto target = std::get<3>(param);

    auto min = minmax[0];
    auto max = minmax[1];
    assert(min <= max);

    auto& shape = form.shape;
    auto& order = form.order;

    std::stringstream test_name;
    test_name << "layers_clamp_" << min << "-" << max
              << "_" << testToString(shape) << "_" << order.toString()
              << "_" << dtype.toString() << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);
    auto layer = om.clamp(input, min, max);
    auto output = om.output(layer);

    ASSERT_TRUE(om.isValid(layer));
    ASSERT_TRUE(om.isValid(om.getSourceOp(layer)));

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

static Form form1({mv::Shape({56, 56, 144, 8}), mv::Order("NCHW")});
static Form form2({mv::Shape({320, 200, 3})   , mv::Order("CHW") });

INSTANTIATE_TEST_CASE_P(demo, layers_clamp,
                        Combine(Values(Minmax({0, 6})),
                                Values(form1, form2),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
