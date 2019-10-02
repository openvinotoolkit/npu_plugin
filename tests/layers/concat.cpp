#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

// FIXME: MCM compiler does not support multiple inputs (yet!)
//        So we assing network's single input be Concat's 1st
//        input, and make constant tensor to feed to Concat as
//        its other inputs (if Concat arity is >1)
#define MULTIPLE_INPUTS 0 // 1=multiple inputs, 0=single input

using Arity = int; // number of input tensors for Concat (must be >0)

using Axis = std::string; // must be either "N", "C", "H", or "W"

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

class layers_concat:
    public LayersTest<std::tuple<Arity, Axis, Form, mv::DType, mv::Target>> {};

TEST_P(layers_concat, dump_blob)
{
    auto param = GetParam();
    auto arity  = std::get<0>(param);
    auto axis   = std::get<1>(param);
    auto form   = std::get<2>(param);
    auto dtype  = std::get<3>(param);
    auto target = std::get<4>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    assert(arity > 0);
    assert(std::string("NCHW").find(axis) != std::string::npos);

    std::stringstream test_name;
    test_name << "layers_concat_" << arity << "_" << axis
              << "_" << testToString(shape) << "_" << order.toString()
              << "_" << dtype.toString() << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(om.input(shape, dtype, order));

    for (int i=0; i < arity-1; i++)
    {
    #if MULTIPLE_INPUTS
        inputs.push_back(om.input(shape, dtype, order));
    #else
        if (dtype == mv::DType("Float16"))
        {
            auto coefs = mv::utils::generateSequence<double>(shape.totalSize());
            inputs.push_back(om.constant(coefs, shape, dtype, order));
        } else
        {
            auto coefs = mv::utils::generateSequence<int64_t>(shape.totalSize());
            inputs.push_back(om.constantInt(coefs, shape, dtype, order));
        }
    #endif
    }

    auto layer = om.concat(inputs, axis);
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

INSTANTIATE_TEST_CASE_P(demo, layers_concat,
                        Combine(Values(Arity(2)),
                                Values(Axis("C")),
                                Values(form1, form2),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
