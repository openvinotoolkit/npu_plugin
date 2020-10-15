#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

enum Func { Conversion=0, Permute=1, Reorder=2 };

static const char *name[] = {"conversion", "permute", "reorder"};

struct ConversionConfig
{
    mv::Shape shape;
    mv::Order order;
    mv::Order order_new;
};

using Config = ConversionConfig;

class layers_conversion:
    public LayersTest<std::tuple<Func, Config, mv::DType, mv::Target>>
{};

TEST_P(layers_conversion, dump_blob)
{
    auto param = GetParam();
    auto func   = std::get<0>(param);
    auto config = std::get<1>(param);
    auto dtype  = std::get<2>(param);
    auto target = std::get<3>(param);

    auto& shape     = config.shape;
    auto& order     = config.order;
    auto& order_new = config.order_new;

    std::string func_name = name[func];

    std::stringstream test_name;
    test_name << "layers_" << func_name
              << "_" << testToString(shape)
              << "_"    << order.toString()
              << "_to_" << order_new.toString()
              << "_" + dtype.toString()
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
        case Conversion:
            layer = om.conversion(input, order_new);
            break;
        case Permute:
            layer = om.permute(input, order_new);
            break;
        case Reorder:
            layer = om.reorder(input, order_new);
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

static Config config4d({mv::Shape({112, 112, 64, 8}), mv::Order("NCHW"), mv::Order("NHWC")});
static Config config3d({mv::Shape({320, 200, 3})    ,  mv::Order("CHW"),  mv::Order("HWC")});

INSTANTIATE_TEST_CASE_P(demo, layers_conversion,
                        Combine(Values(Conversion, Permute, Reorder),
                                Values(config3d, config4d),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
