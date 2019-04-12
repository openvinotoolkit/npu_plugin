#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

#define MULTIPLE_INPUTS 0 // 1=multiple inputs, 0=single input

enum Func { Add, Subtract };

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

class layers_addsub:
    public LayersTest<std::tuple<Func, Form, mv::DType, mv::Target>> {};

TEST_P(layers_addsub, dump_blob)
{
    auto param = GetParam();
    auto func  = std::get<0>(param);
    auto form  = std::get<1>(param);
    auto dtype = std::get<2>(param);
    auto target = std::get<3>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::string func_name = func == Add ? "add": "subtract";
    std::string test_name = "layers_" + func_name + "_" + testToString(shape)
                          + "_" + order.toString() + "_" + dtype.toString()
                          + "_" + testToString(target);
    std::cout << "Test: " << test_name << std::endl;

    testSetName(test_name);

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto data0 = om.input(shape, dtype, order);
#if MULTIPLE_INPUTS
    //---------------------------------------------------
    // MCM compiler does not support multile inputs (yet)
    //---------------------------------------------------
    auto data1 = om.input(shape, dtype, order);
#else
    std::vector<double> data1_coefs =
            mv::utils::generateSequence<double>(shape.totalSize());
    auto data1 = om.constant(data1_coefs, shape, dtype, order);
#endif

    auto addsub = func == Add ? om.add(data0, data1) : om.subtract(data0, data1);
    auto output = om.output(addsub);

    auto compDesc = testGetCompilationDescriptor(unit, target);

    EXPECT_EQ("OK", testSetGenBlob(compDesc));
    EXPECT_EQ("OK", testSetGenDot(compDesc));

    ASSERT_TRUE(unit.loadTargetDescriptor(target));
    ASSERT_TRUE(unit.initialize());

    auto result = unit.run();

    testDumpJson(result);
    testDumpDot();
}

using namespace testing;

static Form form1({mv::Shape({56, 56, 24, 8}), mv::Order("NCHW")});
static Form form2({mv::Shape({320, 200, 3})  , mv::Order("CHW") });

INSTANTIATE_TEST_CASE_P(demo, layers_addsub,
                        Combine(Values(Add, Subtract),
                                Values(form1, form2),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
