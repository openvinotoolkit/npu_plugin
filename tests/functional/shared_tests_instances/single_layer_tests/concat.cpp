// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

#include <iostream>

namespace LayerTestsDefinitions {
    class KmbConcatLayerTest: public ConcatLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        // SkipBeforeLoad() is added because all values for inShapes (except {{10,10,10,10}}) lead to error
        // in mcm-compiler:
        // [Debug  ][VPU][KMB nGraph Parser] Run MCM Compiler
        // kmbFuncTests: kmb-plugin/src/mcmCompiler/src/scheduler/feasible_scheduler.hpp:2200:
        // void mv::lp_scheduler::Feasible_Memory_Schedule_Generator<T, SchedulerTraits, Allocator>::
        // unschedule_op(const mv::lp_scheduler::Feasible_Memory_Schedule_Generator<T, SchedulerTraits, Allocator>::
        // heap_element_t&) [with T = mv::scheduler::Operation_Dag<>;
        // SchedulerTraits = mv::lp_scheduler::scheduler_traits<mv::scheduler::Operation_Dag<> >;
        // Allocator = std::allocator<mv::scheduler::Operation_Dag<> >]: Assertion `itr != op_output_table_.end()' failed.
        // Aborted (core dumped)
        // [Track number: S#49997]
        virtual void SkipBeforeLoad() override {
            int axis;
            std::vector<std::vector<size_t>> inputShapes;
            InferenceEngine::Precision netPrecision;
            InferenceEngine::Precision inPrc, outPrc;
            InferenceEngine::Layout inLayout, outLayout;
            std::string targetName;
            std::tie(axis, inputShapes, netPrecision, inPrc, outPrc, inLayout, outLayout,
                    targetName) = GetParam();

            if (inputShapes.size() == 1) // This is just for inShapes = {{10,10,10,10}}
                return;

            throw LayerTestsUtils::KmbSkipTestException("There is error on step: "
                                                        "[Debug  ][VPU][KMB nGraph Parser] Run MCM Compiler");
        }

        // There is segmentation fault during infer on KMB-board for inShapes_pass_mcm = { {{10, 10, 10, 10}} }.
        // The segfault arises inside function blob_copy_4d_t<Precision::FP32>() in file
        // openvino/inference-engine/src/inference_engine/blob_transform.cpp
        // [Track number: S#49998]
        virtual void SkipBeforeInfer() override {
            throw LayerTestsUtils::KmbSkipTestException("There is \"Segmentation fault\" during infer "
                                                        "on KMB-board revision A");
        }
    };

    TEST_P(KmbConcatLayerTest, CompareWithRefs) {
        Run();
    }

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

InferenceEngine::SizeVector axes = {0, 1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}
};

std::vector<std::vector<std::vector<size_t>>> reshapeTargetShapes = {
    {{20, 10, 10, 10}, {20, 10, 10, 10}, {20, 10, 10, 10}},
    {{20, 20, 10, 10}, {20, 20, 10, 10}, {20, 20, 10, 10}},
    {{20, 20, 20, 10}, {20, 20, 20, 10}, {20, 20, 20, 10}},
    {{20, 20, 20, 20}, {20, 20, 20, 20}, {20, 20, 20, 20}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, KmbConcatLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    KmbConcatLayerTest::getTestCaseName);

// Check parameters from InceptionV3
// This test is just attempt to use parameters other than in CPU-plugin.
// Note: KMB-plugin does not support batch-size > 1.
    InferenceEngine::SizeVector axes_check = {1};

    std::vector<std::vector<std::vector<size_t>>> inShapes_check = {
            {{1, 64, 35, 35}, {1, 64, 35, 35}},
            {{1, 64, 35, 35}, {1, 64, 35, 35}, {1, 96, 35, 35}, {1, 32, 35, 35}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Reshape, KmbConcatLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(axes_check),
                                    ::testing::ValuesIn(inShapes_check),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConcatLayerTest::getTestCaseName);
// end of Check parameters from InceptionV3

}  // namespace
