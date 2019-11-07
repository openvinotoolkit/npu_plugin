#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

#define MULTIPLE_INPUTS 0 // 1=multiple inputs, 0=single input

enum Func { Add=0, Subtract=1, Multiply=2, Divide=3 };

static const char *func_str[] = { "add", "subtract", "multiply", "divide" };

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

class layers_binops:
    public LayersTest<std::tuple<Func, Form, mv::DType, mv::Target>> {};

TEST_P(layers_binops, dump_blob)
{
    auto param = GetParam();
    auto func  = std::get<0>(param);
    auto form  = std::get<1>(param);
    auto dtype = std::get<2>(param);
    auto target = std::get<3>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::string func_name = func_str[func];
    std::string test_name = "layers_" + func_name + "_" + testToString(shape)
                          + "_" + order.toString() + "_" + dtype.toString()
                          + "_" + testToString(target);
    std::cout << "Test: " << test_name << std::endl;

    testSetName(test_name);

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    mv::Data::TensorIterator data0, data1;
    data0 = om.input(shape, dtype, order);
#if MULTIPLE_INPUTS
    //---------------------------------------------------
    // MCM compiler does not support multile inputs (yet)
    //---------------------------------------------------
    auto data1 = om.input(shape, dtype, order);
#else
    if (dtype == mv::DType("Float16"))
    {
        auto coefs = mv::utils::generateSequence<double>(shape.totalSize());
        data1 = om.constant(coefs, shape, dtype, order);
    } else
    {
        auto coefs = mv::utils::generateSequence<int64_t>(shape.totalSize());
        data1 = om.constantInt(coefs, shape, dtype, order);
    }
#endif

    auto binop = func == Add ?
                     om.add({data0, data1}) :
                 func == Subtract ?
                     om.subtract({data0, data1}) :
                 func == Multiply ?
                     om.multiply({data0, data1}) :
    //           func == Divide ?
                     om.divide(data0, data1);
    auto output = om.output(binop);

    ASSERT_TRUE(om.isValid(binop));
    ASSERT_TRUE(om.isValid(om.getSourceOp(binop)));

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

INSTANTIATE_TEST_CASE_P(demo, layers_binops,
                        Combine(Values(Add, Subtract, Multiply, Divide),
                                Values(form1, form2),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
