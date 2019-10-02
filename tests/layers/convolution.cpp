#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

enum Func { Simple=0, Group=1, Depthwise=2 };

static const char *name[] = {"", "group_", "depthwise_"};

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

struct OutputConfig
{
    unsigned chans; // number of output channels
    unsigned group; // must evenly divide chans
};

using KSize    = typename std::array<unsigned short, 2>;
using Stride   = typename std::array<unsigned short, 2>;
using Padding  = typename std::array<unsigned short, 4>;
using Dilation = unsigned;

class layers_convolution:
    public LayersTest<std::tuple<Form, OutputConfig, KSize, Stride, Padding, Dilation,
                                 mv::DType, mv::Target>>
{};

TEST_P(layers_convolution, dump_blob)
{
    auto param = GetParam();
    auto form     = std::get<0>(param);
    auto outconf  = std::get<1>(param);
    auto ksize    = std::get<2>(param);
    auto stride   = std::get<3>(param);
    auto padding  = std::get<4>(param);
    auto dilation = std::get<5>(param);
    auto dtype    = std::get<6>(param);
    auto target   = std::get<7>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    auto& chans = outconf.chans;
    auto& group = outconf.group;

    assert(group > 0 && chans > 0 && chans % group == 0);
    Func func = group == 1     ? Simple :
                group == chans ? Depthwise :
                                 Group;
    std::string func_name = std::string(name[func]) + "convolution";

    std::stringstream test_name;
    test_name << "layers_" << func_name
              << "_" << testToString(shape)
              << "_" << order.toString()
              << "_o" << chans << "x" << group
              << "_k" << ksize[0] << "x" << ksize[1]
              << "_s" << stride[0] << "x" << stride[1]
              << "_p" << padding[0] << "x" << padding[1] <<  "x" << padding[2] << "x" << padding[3]
              << "_d" << dilation
              << "_" + dtype.toString()
              << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);

    auto c_idx = order.toString().find("C");
    assert(c_idx != std::string::npos);
    auto c = shape[(shape.ndims() - 1) - c_idx];

    // shape and order for weights
    mv::Order w_order = mv::Order::getColMajorID(4);
    mv::Shape w_shape({ksize[0], ksize[1], c, chans});

    mv::Data::TensorIterator weights;
    if (dtype == mv::DType("Float16"))
    {
        auto coefs = mv::utils::generateSequence<double>(w_shape.totalSize());
        weights = om.constant(coefs, w_shape, dtype, w_order);
    } else
    {
        auto coefs = mv::utils::generateSequence<int64_t>(w_shape.totalSize());
        weights = om.constantInt(coefs, w_shape, dtype, w_order);
    }

    mv::Data::TensorIterator layer;
    if (func == Depthwise)
    {
        layer = om.depthwiseConv(input, weights, stride, padding, dilation);
    }
    else
    {
        layer = om.conv(input, weights, stride, padding, dilation, group);
    }

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

static Form form1({mv::Shape({112, 112, 64, 8}), mv::Order("NCHW")});
static Form form2({mv::Shape({320, 200, 3, 1})  , mv::Order("NCHW") });

INSTANTIATE_TEST_CASE_P(demo, layers_convolution,
                        Combine(Values(form1, form2),
                                Values(OutputConfig({16,  1}),  //    simple convolution
                                       OutputConfig({16,  4}),  //  groupped convolution
                                       OutputConfig({16, 16})), // depthwise convolution
                                Values(KSize({3, 3})),
                                Values(Stride({2, 2})),
                                Values(Padding({1, 1, 1, 1})),
                                Values(Dilation(1)),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
