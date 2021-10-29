//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux_layer_test.hpp"

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/single_layer/eltwise.hpp>

namespace {

class EltwiseSameInputTest :
        public ov::test::subgraph::EltwiseLayerTest,
        virtual public LayerTestsUtils::VPUXLayerTestsCommon {
};

TEST_P(EltwiseSameInputTest, MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    run();
}

std::vector<std::vector<ov::Shape>> inShapes = {
        {{1, 16, 128, 128}},
};

INSTANTIATE_TEST_CASE_P(smoke, EltwiseSameInputTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                ::testing::Values(CommonTestUtils::OpType::VECTOR),
                                ::testing::Values(ov::element::f16),
                                ::testing::Values(ov::element::f16),
                                ::testing::Values(ov::element::f16),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                ::testing::Values(ov::test::Config{})
                                ),
                        EltwiseSameInputTest::getTestCaseName);

}  // namespace
