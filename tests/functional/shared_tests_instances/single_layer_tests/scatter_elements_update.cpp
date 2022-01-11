// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_elements_update.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbScatterElementsUpdateLayerTest: public ScatterElementsUpdateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbScatterElementsUpdateLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

//indices should not be random value
const std::vector<std::vector<size_t>> _idxValue = {
        {0, 2, 4, 6, 1, 3, 5, 7},
        {1, 0, 4, 6, 2, 3, 7, 5}
};

//inShape, updateShape, axis
const std::vector<axisShapeInShape> _params = {
    {{10,12,15}, {1,2,4}, 0},
};

std::string local_getTestCaseName(const testing::TestParamInfo<scatterElementsUpdateParamsTuple> &obj) {
    axisShapeInShape shapeDescript;
    InferenceEngine::SizeVector indicesValue;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::string targetName;
    std::tie(shapeDescript, indicesValue, inputPrecision, indicesPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "xInShape=" << CommonTestUtils::vec2str(std::get<0>(shapeDescript)) << "_";
    result << "xIdxShape=" << CommonTestUtils::vec2str(std::get<1>(shapeDescript)) << "_";
    result << "xAxis=" << std::get<2>(shapeDescript) << "_";
    result << "xinPrc=" << inputPrecision.name() << "_";
    result << "xidxPrc=" << indicesPrecision.name() << "_";
    result << "xtargetDevice=" << targetName << "_";
    return result.str();
}

INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterElementsUpdate,
        KmbScatterElementsUpdateLayerTest,
        testing::Combine(
           testing::ValuesIn(_params),
           testing::ValuesIn(_idxValue),                      // indices values
           testing::Values(InferenceEngine::Precision::FP16), // input prec
           testing::Values(InferenceEngine::Precision::I32),  // indices prec
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
      #if 1 //=> FATAL gtest-param-util.h "Duplicate parametrized test name..."
       KmbScatterElementsUpdateLayerTest::getTestCaseName
      #else //workaround:
        local_getTestCaseName //WOW: same err
      #endif
);

}  // namespace
