#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "layers.hpp"

using Coords  = unsigned;
using Classes = unsigned;
using DoSoftMax = bool;
using Num = unsigned;
using Mask = std::vector<unsigned>;

struct RegionYoloConfig
{
    mv::Shape shape; // NCHW only!
    Coords    coords;
    Classes   classes;
    DoSoftMax do_softmax;
    Num       num;  // if not do_softmax
    Mask      mask; // if     do_softmax
};

using Config = RegionYoloConfig;

class layers_region_yolo:
    public LayersTest<std::tuple<Config, mv::DType, mv::Target>>
{};

TEST_P(layers_region_yolo, dump_blob)
{
    auto param = GetParam();
    auto config = std::get<0>(param);
    auto dtype  = std::get<1>(param);
    auto target = std::get<2>(param);

    auto  order      = mv::Order("NCHW");

    auto& shape      = config.shape;
    auto  coords     = config.coords;
    auto  classes    = config.classes;
    auto  do_softmax = config.do_softmax;
    auto  num        = config.num;
    auto& mask       = config.mask;

    std::stringstream test_name;
    test_name << "layers_region_yolo"
              << "_" << testToString(shape)
              << "_" << order.toString()
              << "_c" << coords
              << "_cl" << classes
              << "_" << (do_softmax ? "t" : "f")
              << "_" << (do_softmax ? testToString(mask) : std::to_string(num))
              << "_" + dtype.toString()
              << "_" << testToString(target);
    std::cout << "Test: " << test_name.str() << std::endl;

    testSetName(test_name.str());

//  mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input(shape, dtype, order);
    auto layer = om.regionYolo(input, coords, classes, do_softmax, num, mask);
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

static Config config_yolov2({mv::Shape({13, 13, 125, 1}), Coords(4), Classes(20), DoSoftMax(false), Num(5), Mask()});
static Config config_yolov3({mv::Shape({13, 13,  75, 1}), Coords(4), Classes(20), DoSoftMax(true),  Num(0), Mask({0, 1, 2})});

INSTANTIATE_TEST_CASE_P(demo, layers_region_yolo,
                        Combine(Values(config_yolov2, config_yolov3),
                                Values(mv::DType("Float16")),
                                Values(mv::Target::ma2490)));
