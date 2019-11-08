#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

enum Func { ReLU=0, LeakyReLU=1, PReLU=2 };

static const char *name[] = {"relu", "leaky_relu", "prelu"};

struct Form
{
    mv::Shape shape;
    mv::Order order;
};

using Slope = double;

class layers_relu:
    public LayersTest<std::tuple<Func, Form, Slope, mv::DType, mv::Target>>
{};

template<typename T>
static std::vector<T> clone_vec(T value, size_t N)
{
    std::vector<T> vec;
    for (size_t n=0; n < N; n++)
        vec.push_back(value);
    return vec;
}

TEST_P(layers_relu, dump_blob)
{
    auto param = GetParam();
    auto func   = std::get<0>(param);
    auto form   = std::get<1>(param);
    auto slope  = std::get<2>(param);
    auto dtype  = std::get<3>(param);
    auto target = std::get<4>(param);

    auto& shape = form.shape;
    auto& order = form.order;

    std::string func_name = name[func];

    std::stringstream test_name;
    test_name << "layers_" << func_name
              << "_" << testToString(shape)
              << "_" << order.toString()
              << "_" << slope
              << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);

    mv::Data::TensorIterator layer;
    switch (func)
    {
        case ReLU:
            layer = om.relu(input);
            break;
        case LeakyReLU:
            layer = om.leakyRelu(input, slope);
            break;
        case PReLU:
            {
                auto order_str = order.toString();
                auto c_idx = order_str.find("C");
                assert(c_idx != std::string::npos);

                auto ndims = shape.ndims();
                assert(ndims == order_str.length());
                auto c = shape[(ndims - 1) - c_idx];

                mv::Data::TensorIterator slope_tensor;
                if (dtype == mv::DType("Float16"))
                {
                    auto data = clone_vec(static_cast<double>(slope), c);
                    slope_tensor = om.constant(data, {c}, dtype, mv::Order("C"));
                }
                else
                {
                    auto data = clone_vec(static_cast<int64_t>(slope), c);
                    slope_tensor = om.constantInt(data, {c}, dtype, mv::Order("C"));
                }

                layer = om.prelu(input, slope_tensor);
            }
            break;
        default:
            throw "unknown function";
    }

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
static Form form3d({mv::Shape({320, 200,  3   }), mv::Order( "CHW")});
static Form form2d({mv::Shape({           3, 8}), mv::Order("NC")});

INSTANTIATE_TEST_CASE_P(demo, layers_relu,
                        Combine(Values(ReLU, LeakyReLU, PReLU),
                                Values(form4d, form3d, form2d),
                                Values(Slope(0.1)),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
