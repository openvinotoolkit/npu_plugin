//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/plugin/profiling_parser.hpp"

#include <gtest/gtest.h>
#include <ie_common.h>

using namespace vpux;
using namespace vpux::profiling;

TEST(MLIR_ProfilingTaskNameParsing, ValidCases) {
    const std::string upaProfLocation = "pool1?t_Convolution/PROF_13";

    const auto testUPA = [](auto taskName, auto layerType, auto currentPos) -> void {
        auto loc = RawProfilingUPARecord::parseTaskName(taskName);
        EXPECT_EQ(loc.meta.layerType, layerType);
        EXPECT_EQ(loc.prof.currentPos, currentPos);
    };

    const std::string actProfLocation = "pool1?t_Convolution/PROF_8_3_2_0/_cluster_1";
    const std::string actProfLocationClusterId = "pool1?t_Convolution/PROF_8_3_2_0/_profilingBuff_cluster_1";
    const std::string actProfLocationMalformedFusion = "pool1?t_Convolution/relu1?t_relu/PROF_8_3_2_0/_cluster_1";
    const std::string actProfLocationClusterIdMalformedFusion =
            "pool1?t_Convolution?relu1?t_relu/PROF_8_3_2_0/_profilingBuff_cluster_1";

    const auto testACT = [](auto taskName, auto layerType, auto inDdrOffset, auto clusterSize, auto clusterId,
                            auto inClusterOffset, auto tileId) -> void {
        auto loc = RawProfilingACTRecord::parseTaskName(taskName);
        EXPECT_EQ(loc.meta.layerType, layerType);
        EXPECT_EQ(loc.prof.inDdrOffset, inDdrOffset);
        EXPECT_EQ(loc.prof.clusterSize, clusterSize);
        EXPECT_EQ(loc.prof.clusterId, clusterId);
        EXPECT_EQ(loc.prof.inClusterOffset, inClusterOffset);
        EXPECT_EQ(loc.prof.tileId, tileId);
    };

    const std::string dpuProfLocation = "Multiply_8333?t_Convolution/PROF_0_0_2_1-1,1,/_cluster_0";
    const std::string dpuProfLocationMalformedFusion =
            "Multiply_8333?t_Convolution/Add_8400?t_Add/PROF_0_0_2_1-1,1,/_cluster_0";
    const std::string dpuProfLocationMalformedFormat = "Multiply_8333?t_Convolution?PROF_0_0_2_1-1,1,/_cluster_0";

    const auto testDPU = [](auto taskName, auto layerType, int16_t taskId, int16_t memoryId, int32_t maxVariants,
                            int16_t numVariants, int16_t clusterId, int16_t bufferId, int16_t numClusters) -> void {
        auto loc = RawProfilingDPURecord::parseTaskName(taskName, 0);
        EXPECT_EQ(loc.meta.layerType, layerType);
        EXPECT_EQ(loc.prof.taskId, taskId);
        EXPECT_EQ(loc.prof.memoryId, memoryId);
        EXPECT_EQ(loc.prof.maxVariants, maxVariants);
        EXPECT_EQ(loc.prof.numVariants, numVariants);
        EXPECT_EQ(loc.prof.clusterId, clusterId);
        EXPECT_EQ(loc.prof.bufferId, bufferId);
        EXPECT_EQ(loc.prof.numClusters, numClusters);
    };

    const std::string dmaProfLocation = "Multiply_8333?t_Convolution/_cluster_0/PROFTASKEND_82";
    const std::string dmaProfBeginLocation = "Multiply_8333?t_Convolution/_cluster_0/PROFTASKBEGIN";
    const std::string dmaProfLocationMalformedFusion =
            "Multiply_8333?t_Convolution/Add_8400?t_Add/_cluster_0/PROFTASKEND_82";
    const std::string dmaProfBeginLocationMalformedFusion =
            "Multiply_8333?t_Convolution/Add_8400?t_Add/_cluster_0/PROFTASKBEGIN";
    const std::string dmaProfReadLocation = "PROFWORKPOINT_READ";

    const auto testDMA = [](auto taskName, auto layerType, int16_t dmaId) -> void {
        auto loc = RawProfilingDMARecord<DMA27Data_t>::parseTaskName(taskName);
        EXPECT_EQ(loc.meta.layerType, layerType);
        EXPECT_EQ(loc.prof.curDmaId, dmaId);
    };

    const auto testDMABegin = [](auto taskName, auto isBegin) -> void {
        EXPECT_EQ(RawProfilingDMARecord<DMA27Data_t>::isTaskBegin(taskName), isBegin);
    };

    const auto testDMAWorkpointRead = [](auto taskName, auto isWorkpointRead) -> void {
        EXPECT_EQ(RawProfilingDMARecord<DMA27Data_t>::isTaskWorkpointRead(taskName), isWorkpointRead);
    };

    testUPA(upaProfLocation, "Convolution", 13);
    testACT(actProfLocation, "Convolution", 8, 3, 1, 2, 0);
    testACT(actProfLocationClusterId, "Convolution", 8, 3, 1, 2, 0);
    testACT(actProfLocationMalformedFusion, "relu_META_PARSING_ERROR", 8, 3, 1, 2, 0);
    testACT(actProfLocationClusterIdMalformedFusion, "relu_META_PARSING_ERROR", 8, 3, 1, 2, 0);

    testDPU(dpuProfLocation, "Convolution", 0, 0, 1, 1, 0, 0, 2);
    testDPU(dpuProfLocationMalformedFusion, "Add_META_PARSING_ERROR", 0, 0, 1, 1, 0, 0, 2);
    testDPU(dpuProfLocationMalformedFormat, "", 0, 0, 1, 1, 0, 0, 2);

    testDMA(dmaProfLocation, "Convolution", 82);
    testDMA(dmaProfLocationMalformedFusion, "Add_META_PARSING_ERROR", 82);

    testDMABegin(dmaProfBeginLocation, true);
    testDMABegin(dmaProfBeginLocationMalformedFusion, true);
    testDMABegin(dmaProfLocation, false);
    testDMABegin(dmaProfLocationMalformedFusion, false);

    testDMAWorkpointRead(dmaProfReadLocation, true);
    testDMAWorkpointRead(dmaProfLocation, false);
}

template <class ExceptionType>
void testInvalidUPALoc(std::string location) {
    EXPECT_THROW(RawProfilingUPARecord::parseTaskName(location), ExceptionType);
}

template <class ExceptionType>
void testInvalidACTLoc(std::string location) {
    EXPECT_THROW(RawProfilingACTRecord::parseTaskName(location), ExceptionType);
}

template <class ExceptionType>
void testInvalidDPULoc(std::string location) {
    EXPECT_THROW(RawProfilingDPURecord::parseTaskName(location, 0), ExceptionType);
}

template <class ExceptionType>
void testInvalidDMALoc(std::string location) {
    EXPECT_THROW(RawProfilingDMA27Record::parseTaskName(location), ExceptionType);
}

TEST(MLIR_ProfilingTaskNameParsing, InvalidCases) {
    testInvalidUPALoc<InferenceEngine::GeneralError>("pool1?PROF_");
    testInvalidUPALoc<InferenceEngine::GeneralError>("pool1?PROF_13_42");
    testInvalidUPALoc<std::exception>("pool1?PROF_XX");

    testInvalidDPULoc<InferenceEngine::GeneralError>("pool1?PROF_");
    testInvalidDPULoc<std::exception>("pool1?PROF_XX_1_2_3");

    testInvalidACTLoc<InferenceEngine::GeneralError>("pool1?PROF_");
    testInvalidACTLoc<InferenceEngine::GeneralError>("pool1?PROF_13_42");
    testInvalidUPALoc<std::exception>("pool1?PROF_XX");

    testInvalidACTLoc<InferenceEngine::GeneralError>("pool1?PROF_");
    testInvalidACTLoc<InferenceEngine::GeneralError>("pool1?PROF_13_42");
    testInvalidUPALoc<std::exception>("pool1?PROF_XX");
}
