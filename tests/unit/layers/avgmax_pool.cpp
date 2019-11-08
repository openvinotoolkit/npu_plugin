#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

enum Func { AveragePooling, MaxPooling };

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

using KSize        = typename std::array<unsigned short, 2>;
using Stride       = typename std::array<unsigned short, 2>;
using Padding      = typename std::array<unsigned short, 4>;
using ExcludePad   = bool;
using AutoPad      = std::string;
using RoundingType = std::string;

class layers_avgmax_pool:
    public LayersTest<std::tuple<Func, Form, KSize, Stride, Padding,
                                 ExcludePad, AutoPad, RoundingType ,
                                 mv::DType, mv::Target>>
{};

TEST_P(layers_avgmax_pool, dump_blob)
{
    auto param = GetParam();
    auto func          = std::get<0>(param);
    auto form          = std::get<1>(param);
    auto ksize         = std::get<2>(param);
    auto stride        = std::get<3>(param);
    auto padding       = std::get<4>(param);
    auto exclude_pad   = std::get<5>(param);
    auto auto_pad      = std::get<6>(param);
    auto rounding_type = std::get<7>(param);
    auto dtype         = std::get<8>(param);
    auto target        = std::get<9>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::string func_name = func == AveragePooling ? "average_pooling": "max_pooling";

    std::stringstream test_name;
    test_name << "layers_" << func_name
              << "_" << testToString(shape)
              << "_" << order.toString()
              << "_k" << ksize[0] << "x" << ksize[1]
              << "_s" << stride[0] << "x" << stride[1]
              << "_p" << padding[0] << "x" << padding[1] << "x" << padding[2] << "x" << padding[3]
              << "_" << (exclude_pad ? "t": "f")
              << "_" << (auto_pad == "" ? "none": auto_pad)
              << "_" << rounding_type
              << "_" << dtype.toString()
              << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);
    auto pool = func == AveragePooling ?
                om.averagePool(input, ksize, stride, padding, exclude_pad, auto_pad, rounding_type) :
                    om.maxPool(input, ksize, stride, padding, exclude_pad, auto_pad, rounding_type);
    auto output = om.output(pool);

    ASSERT_TRUE(om.isValid(pool));
    ASSERT_TRUE(om.isValid(om.getSourceOp(pool)));

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

static Form form1({mv::Shape({112, 112, 64, 8}), mv::Order("NCHW")});
static Form form2({mv::Shape({320, 200, 3, 1})  , mv::Order("NCHW") });

INSTANTIATE_TEST_CASE_P(demo, layers_avgmax_pool,
                        Combine(Values(AveragePooling, MaxPooling),
                                Values(form1, form2),
                                Values(KSize({3, 3})),
                                Values(Stride({2, 2})),
                                Values(Padding({1, 1, 1, 1})),
                                Values(ExcludePad(true)),
                                Values(AutoPad("")),
                                Values(RoundingType("floor")),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
