#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

enum Func { Bias=0, Scale=1 };

static const char *name[] = { "bias", "scale" };

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

class layers_bias_scale:
    public LayersTest<std::tuple<Func, Form, mv::DType, mv::Target>> {};

TEST_P(layers_bias_scale, dump_blob)
{
    auto param = GetParam();
    auto func   = std::get<0>(param);
    auto form   = std::get<1>(param);
    auto dtype  = std::get<2>(param);
    auto target = std::get<3>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::string func_name = name[func];

    std::string test_name = "layers_" + func_name
                          + "_" + testToString(shape) + "_" + order.toString()
                          + "_" + dtype.toString() + "_" + testToString(target);
    std::cout << "Test: " << test_name << std::endl;

    testSetName(test_name);

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);

    // which index is for channels (for "C")
    auto C_idx = order.toString().find("C");
    assert(C_idx != std::string::npos);

    // note: opposite enumeration of dimensions
    auto C = shape[(shape.ndims() - 1) - C_idx];

    mv::Data::TensorIterator weights;
    if (dtype == mv::DType("Float16"))
    {
        auto coefs = mv::utils::generateSequence<double>(C);
        weights = om.constant(coefs, {C}, dtype, mv::Order("C"));
    } else
    {
        auto coefs = mv::utils::generateSequence<int64_t>(C);
        weights = om.constantInt(coefs, {C}, dtype, mv::Order("C"));
    }

    auto bias = om.bias(input, weights);
    auto output = om.output(bias);

    ASSERT_TRUE(om.isValid(bias));
    ASSERT_TRUE(om.isValid(om.getSourceOp(bias)));

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

static Form form1({mv::Shape({56, 56, 24, 8}), mv::Order("NCHW")});
static Form form2({mv::Shape({320, 200, 3})  , mv::Order("CHW") });

INSTANTIATE_TEST_CASE_P(demo, layers_bias_scale,
                        Combine(Values(Bias, Scale),
                                Values(form1, form2),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
