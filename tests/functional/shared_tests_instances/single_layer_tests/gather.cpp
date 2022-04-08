// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <random>
#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include <vpux/vpux_plugin_config.hpp>

namespace LayerTestsDefinitions {

class KmbGatherLayerTest: public GatherLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

    void SkipBeforeLoad() override {
        std::vector<size_t> inputShape;
        std::string device;
        std::tie(std::ignore, std::ignore, std::ignore, inputShape, std::ignore, std::ignore, std::ignore,
                std::ignore, std::ignore, device) = GetParam();
        
        if (inputShape.size() != 4) {
            throw LayerTestsUtils::KmbSkipTestException("Runtime only supports 4D input shape");
        }

        if (device == "EMULATOR") {
            throw LayerTestsUtils::KmbSkipTestException(
                "Emulator device does not support Gather with I32 second input");
        }

#if (defined(WINNT) || defined(_WIN32) || defined(_WIN64))
        // Test on Gather with MCM-compiler on dKMB Windows fails with error "zeCommandQueueExecuteCommandLists result: 0x70000001"        
        // [Track number: S#36265]
        if (configuration["VPUX_COMPILER_TYPE"] == "MCM") {
            throw LayerTestsUtils::KmbSkipTestException("Firmware error: Failed to parse blob");
        }
#endif
    }
};

class KmbGather7LayerTest: public Gather7LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

    void SkipBeforeLoad() override {
        auto inputRank = std::get<0>(GetParam()).size();
        auto indexRank = std::get<1>(GetParam()).size();
        auto batchDims = std::get<1>(std::get<2>(GetParam()));

        if (inputRank != 4) {
            throw LayerTestsUtils::KmbSkipTestException("Gather7 only supports 4D input shape, inRank = " + std::to_string(inputRank));
        }

        auto outRank = inputRank + indexRank - 1 - batchDims;
        if (outRank != 4){
            throw LayerTestsUtils::KmbSkipTestException("Gather7 only supports 4D output shape, outRank = " + std::to_string(outRank));
        }
    }
};

TEST_P(KmbGatherLayerTest, CompareWithRefs) {
   // Enable NCHW layout
    core->SetConfig({{VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT), CONFIG_VALUE(YES)}},
                      LayerTestsUtils::testPlatformTargetDevice);
    Run();
}

TEST_P(KmbGatherLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbGather7LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes = {
        //std::vector<size_t>{10, 20, 30, 40},
          std::vector<size_t>{ 5,  6,  7,  8},
};

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes = {
        std::vector<size_t>{4},
        // std::vector<size_t>{2, 2}  //  Only 1D shape for indices is supported
};

const std::vector<int> axes = {0, 1, 2, 3, /*-1*/};  // Only positive axis value is supported

const auto params = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

// nGraph parser doesn't contain specific gather parser
// [Track number: S#40603]
INSTANTIATE_TEST_SUITE_P(
        smoke_Gather1,
        KmbGatherLayerTest,
        params,
        KmbGatherLayerTest::getTestCaseName
);

}  // namespace


namespace { //conformance scenarios

const auto genParams(const std::vector<size_t> inputShape, const int axis, const size_t idxNum)
{
  std::vector<int> _indices(idxNum, 0);

  if(axis>=inputShape.size()){
     std::cout << "error: axis=" << axis << " out of range, ";
     std::cout << "valid range = [0.." << inputShape.size()-1 << "]" << std::endl;
     abort();
  }

  // Initialize indices within valid range
  const size_t max = inputShape[axis];
  std::default_random_engine gen(123);
  std::uniform_int_distribution<int> distrib(0, max-1);
  for (size_t i = 0; i < _indices.size(); i++) {
     _indices[i] = distrib(gen);
  }

  return testing::Combine(
        testing::Values(_indices),
        testing::Values(std::vector<size_t>{idxNum}),
        testing::Values(axis),
        testing::Values(inputShape),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
  );
}

#define GEN_TEST(no,inputShape,axis,numIndices)\
INSTANTIATE_TEST_CASE_P( \
        conform_Gather1_ ## no, \
        KmbGatherLayerTest, \
        genParams(inputShape,axis,numIndices),\
        KmbGatherLayerTest::getTestCaseName)


GEN_TEST( 0,(std::vector<size_t>{ 10, 20, 30, 40}), 2,   4); //=> {10,20,4,40}
GEN_TEST( 1,(std::vector<size_t>{ 32,  3,  3,  3}), 0,  27); //=> {27,3,3,3}
GEN_TEST( 2,(std::vector<size_t>{ 32,  1,  3,  3}), 0,  27); //=> {27,1,3,3}
GEN_TEST( 3,(std::vector<size_t>{ 16, 32,  3,  3}), 1,  27); //=> {16,27,3,3}
GEN_TEST( 4,(std::vector<size_t>{ 96, 16,  1,  1}), 0,  95); //=> {95,16,1,1}
GEN_TEST( 5,(std::vector<size_t>{ 24, 96,  1,  1}), 1,  95); //=> {24,95,1,1}
GEN_TEST( 6,(std::vector<size_t>{144, 24,  1,  1}), 0, 143); //=> {143,24,1,1}
GEN_TEST( 7,(std::vector<size_t>{144,  1,  3,  3}), 0, 143); //=> {143,1,3,3}
GEN_TEST( 8,(std::vector<size_t>{ 24,144,  1,  1}), 1, 143); //=> {24,143,1,1}
GEN_TEST( 9,(std::vector<size_t>{192, 32,  1,  1}), 0, 191); //=> {191,32,1,1}
GEN_TEST(10,(std::vector<size_t>{ 32,192,  1,  1}), 1, 191); //=> {32,191,1,1}
GEN_TEST(11,(std::vector<size_t>{384,  1,  3,  3}), 0, 380); //=> {380,1,3,3}
GEN_TEST(12,(std::vector<size_t>{576,  1,  3,  3}), 0, 574); //=> {574,1,3,3}
GEN_TEST(13,(std::vector<size_t>{576,  1,  3,  3}), 0, 571); //=> {571,1,3,3}
GEN_TEST(14,(std::vector<size_t>{960,  1,  3,  3}), 0, 954); //=> {954,1,3,3}
GEN_TEST(15,(std::vector<size_t>{960,  1,  3,  3}), 0, 959); //=> {959,1,3,3}
GEN_TEST(16,(std::vector<size_t>{  2,  64,  1, 1}), 0, 128); //=> {128,64,1,1}
GEN_TEST(17,(std::vector<size_t>{  2,  64,  1, 1}), 1, 128); //=> {2,128,1,1}

}  // namespace

namespace { // opset7::Gather tests

#define GEN7_TEST(no,inputShape,indicesShape,axis,batch_dims) \
INSTANTIATE_TEST_CASE_P( \
        smoke_Gather7_ ## no, \
        KmbGather7LayerTest, \
        testing::Combine( \
          testing::Values(std::vector<size_t>inputShape), \
          testing::Values(std::vector<size_t>indicesShape), \
          testing::Values(std::tuple<int,int>{axis,batch_dims}), \
          testing::Values(InferenceEngine::Precision::FP16), \
          testing::Values(InferenceEngine::Precision::FP16), \
          testing::Values(InferenceEngine::Precision::FP16), \
          testing::Values(InferenceEngine::Layout::ANY), \
          testing::Values(InferenceEngine::Layout::ANY), \
          testing::Values(LayerTestsUtils::testPlatformTargetDevice)), \
        KmbGather7LayerTest::getTestCaseName )

GEN7_TEST(0, ({3,5,1,1}),     ({3,2}), 1, 1);
GEN7_TEST(1, ({4,3,5,1}),     ({4,4}), 2, 1);
GEN7_TEST(2, ({3,2,1,1}),     ({3,2}), 1, 1);
GEN7_TEST(3, ({2,2,5,1}),   ({2,2,3}), 2, 2);
GEN7_TEST(4, ({2,1,5,4}),     ({2,3}), 2, 1);
GEN7_TEST(5, ({2,5,2,1}),   ({2,2,3}), 1, 1);
GEN7_TEST(6, ({2,5,1,1}),     ({2,3}), 1, 1);

}  // namespace
