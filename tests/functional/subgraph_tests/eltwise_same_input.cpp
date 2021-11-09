// //
// // Copyright Intel Corporation.
// //
// // LEGAL NOTICE: Your use of this software and any required dependent software
// // (the "Software Package") is subject to the terms and conditions of
// // the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// // which may also include notices, disclaimers, or license terms for
// // third party or open source software included in or with the Software Package,
// // and your use indicates your acceptance of all such terms. Please refer
// // to the "third-party-programs.txt" or other similarly-named text file
// // included with the Software Package for additional details.
// //

// #include "kmb_layer_test.hpp"

// #include <ngraph_functions/builders.hpp>
// #include <ngraph_functions/utils/ngraph_helpers.hpp>
// #include <shared_test_classes/base/layer_test_utils.hpp>
// #include <shared_test_classes/single_layer/eltwise.hpp>

// namespace {

// class EltwiseSameInputTest :
//         public LayerTestsDefinitions::EltwiseLayerTest,
//         virtual public LayerTestsUtils::KmbLayerTestsCommon {
// protected:
//     void SetUp() override {
//         //std::vector<std::vector<size_t>> inputShapes;
//         std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> inputShapes;
//         InferenceEngine::Precision netPrecision;
//         ngraph::helpers::EltwiseTypes eltwiseType;
//         std::tie(inputShapes, eltwiseType, std::ignore, std::ignore, netPrecision, inPrc, outPrc, inLayout,
//                  targetDevice, configuration) = GetParam();

//         if (inputShapes.second.size() != 1) {
//             IE_THROW() << "Incorrect number of input shapes";
//         }

//         const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

//         const auto params = ngraph::builder::makeParams(ngPrc, {inputShapes.second.front().front()});
//         const auto eltwise = ngraph::builder::makeEltwise(params[0], params[0], eltwiseType);
//         function = std::make_shared<ngraph::Function>(eltwise, params, "Eltwise");
//     }
// };

// TEST_P(EltwiseSameInputTest, MLIR_HW) {
//     useCompilerMLIR();
//     setReferenceHardwareModeMLIR();
//     Run();
// }

// std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapes = {
//        { {}, {{{1, 16, 128, 128}}} }
// };

// //INSTANTIATE_TEST_SUITE_P(smoke, EltwiseSameInputTest,
// INSTANTIATE_TEST_CASE_P(smoke, EltwiseSameInputTest,
//                         ::testing::Combine(                                                     //
//                                 ::testing::ValuesIn(inShapes),                                  //
//                                 ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),          //
//                                 ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),  //
//                                 ::testing::Values(CommonTestUtils::OpType::VECTOR),             //
//                                 ::testing::Values(InferenceEngine::Precision::FP16),            //
//                                 ::testing::Values(InferenceEngine::Precision::FP16),            //
//                                 ::testing::Values(InferenceEngine::Precision::FP16),            //
//                                 ::testing::Values(InferenceEngine::Layout::ANY),                //
//                                 ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),   //
//                                 ::testing::Values(std::map<std::string, std::string>{})         //
//                                 ),
//                         EltwiseSameInputTest::getTestCaseName);

// }  // namespace
