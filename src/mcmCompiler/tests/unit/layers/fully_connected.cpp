#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

using OutputChannels = unsigned;

class layers_fully_connected:
    public LayersTest<std::tuple<Form, OutputChannels, mv::DType, mv::Target>>
{};

TEST_P(layers_fully_connected, dump_blob)
{
    auto param = GetParam();
    auto form     = std::get<0>(param);
    auto outchans = std::get<1>(param);
    auto dtype    = std::get<2>(param);
    auto target   = std::get<3>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::stringstream test_name;
    test_name << "layers_fully_connected"
              << "_" << testToString(shape)
              << "_" << order.toString()
              << "_" << outchans
              << "_" + dtype.toString()
              << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);

    auto ndims = shape.ndims();
    auto order_str = order.toString();
    assert(ndims == order_str.length());

    std::string         w_order_str; // weights order
    std::vector<size_t> w_shape_vec; // weights shape
    for (unsigned i=0; i < ndims; i++)
    {
        const char d = order_str[(ndims - 1) - i];
        assert(std::string("NCHW").find(d) != std::string::npos);
        if (d != 'N') {
            w_order_str = d + w_order_str;
            w_shape_vec.push_back(shape[i]);
        }
    }

    // FIXME: register a right name for output channels
    w_order_str = "C" + w_order_str; // output channels
    w_shape_vec.push_back(outchans);

    mv::Shape w_shape(w_shape_vec);
#if 0
    mv::Order w_order(w_order_str);
#else
    // FIXME: register a "right" order for the weights in MCM compiler
    mv::Order w_order(mv::Order::getColMajorID(w_order_str.length()));
#endif

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

    auto layer = om.fullyConnected(input, weights);
    auto layerOp = om.getSourceOp(layer);

    auto output = om.output(layer);

    ASSERT_TRUE(om.isValid(layer));
    ASSERT_TRUE(om.isValid(layerOp));

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

static Form form4d({mv::Shape({112, 112, 64, 8}), mv::Order("NCHW")});
static Form form3d({mv::Shape({320, 200, 3})    ,  mv::Order("CHW")});
static Form form2d({mv::Shape({128, 8})         , mv::Order("NC")});

INSTANTIATE_TEST_CASE_P(demo, layers_fully_connected,
                        Combine(Values(form4d, form3d, form2d),
                                Values(OutputChannels(16)),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
