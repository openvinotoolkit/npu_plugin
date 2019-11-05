#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

struct ReshapeConfig
{
    mv::Shape shape;
    mv::Order order;
    mv::Shape shape_new;
  std::string order_new;
};

using Config = ReshapeConfig;

class layers_reshape:
    public LayersTest<std::tuple<Config, mv::DType, mv::Target>>
{};

TEST_P(layers_reshape, dump_blob)
{
    auto param = GetParam();
    auto config = std::get<0>(param);
    auto dtype  = std::get<1>(param);
    auto target = std::get<2>(param);

    auto& shape     = config.shape;
    auto& order     = config.order;
    auto& shape_new = config.shape_new;
    auto& order_new = config.order_new;

    std::stringstream test_name;
    test_name << "layers_reshape"
              << "_" << testToString(shape)
              << "_" << order.toString()
              << "_to_" << testToString(shape_new)
              << (order_new == "" ? "" : "_" + order_new)
              << "_" + dtype.toString()
              << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);
    auto layer = om.reshape(input, shape_new, order_new);
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

static Config conf4d({mv::Shape({320, 200, 3, 8}), mv::Order("NCHW"), mv::Shape({160, 100, 12, 8}), ""});
static Config conf3d({mv::Shape({320, 200, 3}),     mv::Order("CHW"), mv::Shape({160, 100, 12}),    ""});
static Config conf2d({mv::Shape({320, 200}),         mv::Order("HW"), mv::Shape({160, 400}),        ""});

static Config conf43({mv::Shape({320, 200, 3, 8}), mv::Order("NCHW"), mv::Shape({320, 200, 24}), "CHW"});
static Config conf32({mv::Shape({320, 200, 3}),     mv::Order("CHW"), mv::Shape({320, 600}),      "HW"});

INSTANTIATE_TEST_CASE_P(demo, layers_reshape,
                        Combine(Values(conf4d, conf3d, conf43, conf32),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
