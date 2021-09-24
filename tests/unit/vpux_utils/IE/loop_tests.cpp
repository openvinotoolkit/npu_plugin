//
// Copyright 2020-2021 Intel Corporation.
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

#include "vpux/utils/IE/loop.hpp"
#include <chrono>

#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_arena.h"
#include "tbb/task_scheduler_observer.h"

#include <ie_parallel.hpp>
#include <gtest/gtest.h>

using namespace vpux;
using namespace std::chrono;

class LoopTests : public testing::Test {
public:
    void SetUp() override;

protected:
    const size_t sizeBuffer = 8000*1000*4;
    std::vector<uint8_t> inData;
    std::vector<float> outData1;
    std::vector<float> outData2;
};

void LoopTests::SetUp() {
    inData.resize(sizeBuffer);
    for (auto& elem: inData) {
        elem = std::rand()%(std::numeric_limits<uint8_t>::max()+1);
    }
    outData1.resize(sizeBuffer);
    outData2.resize(sizeBuffer);
}

TEST_F(LoopTests, LoopAndBlockedLoopProvideTheSameResult) {
    const auto inPtr = inData.data();
    auto out1Ptr = outData1.data();
    loop_1d(LoopExecPolicy::Parallel, sizeBuffer, [inPtr, out1Ptr](int64_t index) {
        out1Ptr[index] = static_cast<float>(inPtr[index]);
    });

    auto out2Ptr = outData2.data();
    blocked_loop_1d(LoopExecPolicy::Parallel, sizeBuffer, [inPtr, out2Ptr](size_t startIndex, size_t endIndex) {
        for (auto index = startIndex; index <= endIndex; ++index) {
            out2Ptr[index] = static_cast<float>(inPtr[index]);
        }
    });

    ASSERT_EQ(outData1, outData2);
}

TEST_F(LoopTests, BlockedLoopIsFasterThenLoop) {
    const auto inPtr = inData.data();
    auto out1Ptr = outData1.data();

    //warm up
    loop_1d(LoopExecPolicy::Parallel, sizeBuffer, [inPtr, out1Ptr](int64_t index) {
        out1Ptr[index] = static_cast<float>(inPtr[index]);
    });

    auto tBegin = high_resolution_clock::now();
    loop_1d(LoopExecPolicy::Parallel, sizeBuffer, [inPtr, out1Ptr](int64_t index) {
        out1Ptr[index] = static_cast<float>(inPtr[index]);
    });
    auto tEnd = high_resolution_clock::now();
    const auto loopTime = duration_cast<microseconds>(tEnd-tBegin).count();

    tBegin = high_resolution_clock::now();
    InferenceEngine::parallel_for_v2(sizeBuffer, [inPtr, out1Ptr](int64_t index) {
        out1Ptr[index] = static_cast<float>(inPtr[index]);
    });
    tEnd = high_resolution_clock::now();
    const auto loopV2Time = duration_cast<microseconds>(tEnd-tBegin).count();

    auto out2Ptr = outData2.data();
    tBegin = high_resolution_clock::now();
    blocked_loop_1d(LoopExecPolicy::Parallel, sizeBuffer, [inPtr, out2Ptr](size_t startIndex, size_t endIndex) {
        for (auto index = startIndex; index <= endIndex; ++index) {
            out2Ptr[index] = static_cast<float>(inPtr[index]);
        }
    });
    tEnd = high_resolution_clock::now();
    const auto blockedLoopTime = duration_cast<microseconds>(tEnd-tBegin).count();

    tBegin = high_resolution_clock::now();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sizeBuffer), [&](tbb::blocked_range<size_t> r) {
        for (size_t index=r.begin(); index<r.end(); ++index) {
            out2Ptr[index] = static_cast<float>(inPtr[index]);
        }
    });
    tEnd = high_resolution_clock::now();
    const auto maximTime = duration_cast<microseconds>(tEnd-tBegin).count();

    std::cout << "Loop took " << loopTime << " us" << std::endl;
    std::cout << "LoopV2 took " << loopV2Time << " us" << std::endl;
    std::cout << "Blocked loop took " << blockedLoopTime << " us" << std::endl;
    std::cout << "Maxim time loop took " << maximTime << " us" << std::endl;
    ASSERT_LT(blockedLoopTime, loopTime);
    ASSERT_EQ(outData1, outData2);
}
