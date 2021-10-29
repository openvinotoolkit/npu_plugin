////
//// Copyright Intel Corporation.
////
//// LEGAL NOTICE: Your use of this software and any required dependent software
//// (the "Software Package") is subject to the terms and conditions of
//// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
//// which may also include notices, disclaimers, or license terms for
//// third party or open source software included in or with the Software Package,
//// and your use indicates your acceptance of all such terms. Please refer
//// to the "third-party-programs.txt" or other similarly-named text file
//// included with the Software Package for additional details.
////
//
//#include "kmb_layer_test.hpp"
//
//#include <ngraph_functions/builders.hpp>
//#include <ngraph_functions/utils/ngraph_helpers.hpp>
//#include <shared_test_classes/base/layer_test_utils.hpp>
//#include <shared_test_classes/single_layer/eltwise.hpp>
//
//namespace {
//
//class EltwiseSameInputTest :
//        public ov::test::subgraph::EltwiseLayerTest,
//        virtual public LayerTestsUtils::KmbLayerTestsCommon {
//protected:
//    void TearDown() override {
//        ov::test::subgraph::EltwiseLayerTest::TearDown();
//    }
//    void SetUp() override {
//        std::vector<ov::test::InputShape> inputShapes;
//        ov::test::ElementType netPrecision;
//        ngraph::helpers::EltwiseTypes eltwiseType;
//        std::tie(inputShapes,
//                 eltwiseType, std::ignore, std::ignore, netPrecision, std::ignore, std::ignore) = GetParam();
//
//        if (inputShapes.front().second.size() != 1) {
//            IE_THROW() << "Incorrect number of input shapes";
//        }
//
//        const auto params = ngraph::builder::makeParams(netPrecision, {inputShapes.front().second.front()});
//        const auto eltwise = ngraph::builder::makeEltwise(params[0], params[0], eltwiseType);
//        LayerTestsUtils::KmbLayerTestsCommon::function = std::make_shared<ngraph::Function>(eltwise, params, "Eltwise");
//    }
//};
//
//TEST_P(EltwiseSameInputTest, MLIR_HW) {
//    useCompilerMLIR();
//    setReferenceHardwareModeMLIR();
//    Run();
//}
//
//std::vector<std::vector<ov::Shape>> inShapes = {
//       {{1, 16, 128, 128}}
//};
//
//INSTANTIATE_TEST_CASE_P(smoke, EltwiseSameInputTest,
//                        ::testing::Combine(                                                                    //
//                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)), //
//                                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),                         //
//                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),                 //
//                                ::testing::Values(CommonTestUtils::OpType::VECTOR),                            //
//                                ::testing::Values(ov::element::f16),                                           //
//                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),                  //
//                                ::testing::Values(std::map<std::string, std::string>{})                        //
//                                ),
//                        EltwiseSameInputTest::getTestCaseName);
//
//}  // namespace
