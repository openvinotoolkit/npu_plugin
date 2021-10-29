//// Copyright (C) 2020 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "single_layer_tests/softmax.hpp"
//
//#include <vector>
//
//#include "kmb_layer_test.hpp"
//
//namespace ov {
//namespace test {
//namespace subgraph {
//
//class KmbSoftMaxLayerTest: public SoftMaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
//    void TearDown() override {
//        SubgraphBaseTest::TearDown();
//    }
//
//    void SkipBeforeLoad() override {
//        ov::test::ElementType inPrecision, outPrecision;
//        ov::test::InputShape inShapes;
//        size_t axisInd;
//        std::tie(std::ignore,
//                 inPrecision, outPrecision,
//                 inShapes,
//                 axisInd,
//                 std::ignore,
//                 std::ignore) = GetParam();
//
//        if (isCompilerMCM()) {
//            // [Track number: S#44702]
//            if (inPrecision == ov::element::f32 || outPrecision == ov::element::f32) {
//                throw LayerTestsUtils::KmbSkipTestException("SoftMax with FP32 input/output hangs on graph loading");
//            }
//
//            // [Track number: S#40296]
//            for (const auto& shape : inShapes.second) {
//                if (shape.at(axisInd) == 1) {
//                    throw LayerTestsUtils::KmbSkipTestException("SoftMax over dim==1 fails during blob parsing");
//                }
//            }
//        }
//    }
//};
//
//TEST_P(KmbSoftMaxLayerTest, CompareWithRefs) {
//    Run();
//}
//
//TEST_P(KmbSoftMaxLayerTest, CompareWithRefs_MLIR) {
//    useCompilerMLIR();
//    Run();
//}
//
//}  // namespace subgraph
//}  // namespace test
//}  // namespace ov
//
//using namespace ngraph::helpers;
//using namespace ov::test::subgraph;
//
//namespace {
//
//const std::vector<ov::test::ElementType> netPrecisions = {
//        ov::element::f32,
//};
//
//const std::vector<ov::Shape> inShapes2D = {
//        {1, 100},
//        {100, 1},
//        {10, 10},
//};
//
//const std::vector<size_t> axis2D = {
//        0, 1
//};
//
//const std::vector<ov::test::ElementType> inputPrecisions = {
//    ov::element::u8,
//    ov::element::f16,
//    ov::element::f32,
//};
//
//const auto params2D = testing::Combine(
//    testing::ValuesIn(netPrecisions),
//    testing::ValuesIn(inputPrecisions),
//    testing::Values(ov::element::undefined),
//    testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes2D)),
//    testing::ValuesIn(axis2D),
//    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
//    testing::Values(std::map<std::string, std::string>())
//);
//
//INSTANTIATE_TEST_CASE_P(
//    smoke_SoftMax2D,
//    KmbSoftMaxLayerTest,
//    params2D,
//    SoftMaxLayerTest::getTestCaseName
//);
//
//const std::vector<ov::Shape> inShapes4D = {
//    {1, 2, 204, 62},
//    {1, 12, 2, 1444},
//    {1, 2, 72, 10},
//    {1, 4, 1, 1},
//    {1, 1000, 1, 1},
//    {300, 21, 1, 1},
//};
//
//const std::vector<size_t> axis4D = {0, 1, 2, 3};
//
//const auto params4D = testing::Combine(
//    testing::ValuesIn(netPrecisions),
//    testing::ValuesIn(inputPrecisions),
//    testing::Values(ov::element::undefined),
//    testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes4D)),
//    testing::ValuesIn(axis4D),
//    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
//    testing::Values(std::map<std::string, std::string>())
//);
//
//INSTANTIATE_TEST_CASE_P(
//        smoke_SoftMax4D,
//        KmbSoftMaxLayerTest,
//        params4D,
//        SoftMaxLayerTest::getTestCaseName
//);
//
//}  // namespace
