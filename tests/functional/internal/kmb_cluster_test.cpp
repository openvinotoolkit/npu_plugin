//
// Copyright 2021 Intel Corporation.
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

#include "test_model/kmb_test_base.hpp"
#include <condition_variable>
#include <fstream>
#include "common/functions.h"

class KmbClusterTest : public KmbNetworkTestBase {
public:
    int32_t runTest(const TestNetworkDesc& netDesc, const std::string& netFileName);
};

int32_t KmbClusterTest::runTest(const TestNetworkDesc& netDesc, const std::string& netFileName) {
    const auto blobFileName = vpu::formatString("%v/%v.net", KmbTestBase::DUMP_PATH, netFileName);
    if (KmbTestBase::RUN_COMPILER) {
        ExecutableNetwork exeNet = getExecNetwork(netDesc);
        exeNet.Export(blobFileName);
    }

    constexpr size_t MAX_ITER_COUNT = 3000;
    constexpr size_t MAX_REQ_COUNT = 8;
    std::condition_variable condVar;
    size_t iterCount = 0;

    if (KmbTestBase::RUN_INFER) {
        // Skip if blob was not generated on host
        std::ifstream file(blobFileName, std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            std::stringstream str;
            str << "importNetwork() failed. Cannot open file " << blobFileName;
            throw import_error(str.str());
        }
        ExecutableNetwork importedNet = core->ImportNetwork(blobFileName, DEVICE_NAME, {});
        std::vector<InferenceEngine::InferRequest> inferRequestVec;
        for (size_t reqId = 0; reqId < MAX_REQ_COUNT; reqId++) {
            inferRequestVec.push_back(importedNet.CreateInferRequest());
            inferRequestVec.at(reqId).SetCompletionCallback([reqId, &inferRequestVec, &condVar, &iterCount, &MAX_ITER_COUNT] {
                iterCount++;
                if (iterCount < MAX_ITER_COUNT) {
                    inferRequestVec.at(reqId).StartAsync();
                } else {
                    condVar.notify_one();
                }
            });
        }

        const auto msBeforeRequest = std::chrono::steady_clock::now();
        for (auto& inferRequest : inferRequestVec) {
            inferRequest.StartAsync();
        }

        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&] { return iterCount >= MAX_ITER_COUNT; });
        const auto msAfterRequest = std::chrono::steady_clock::now();

        const auto requestDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(msAfterRequest - msBeforeRequest);
        return requestDurationMs.count();
    }

    return 0;
}

struct ClusterTestParams final {
    ClusterTestParams(const std::string& netName, const std::string& numClusters) : _netName(netName), _numClusters(numClusters) {}
    const std::string _netName;
    const std::string _numClusters;
};

std::ostream& operator<<(std::ostream& os, const ClusterTestParams& p) {
    vpu::formatPrint(os, "[net name: %s, clusters: %s]", p._netName, p._numClusters);
    return os;
}

class KmbClusterTestWithParams : public KmbClusterTest, public testing::WithParamInterface<ClusterTestParams> {};

TEST_P(KmbClusterTestWithParams, precommit_checkInferTime) {
#ifdef _WIN32
    // FIXME [Track number: E#6518]
    SKIP() << "Throws an exception on the second runTest call";
#endif
    if (PlatformEnvironment::PLATFORM == "3400_A0") {
        // FIXME [Track number: E#10416]
        SKIP() << "MCM Compiler error: Failed to pass runtime simulation";
    }
    const auto& p = GetParam();
    const std::string net_path = "ADK3/ModelE_INT8/ModelE_INT8.xml";
    constexpr bool EXPERIMENTAL = true;

    const std::string netName = p._netName;
    const std::string clusters = p._numClusters;
    try {
        const auto timeMs = runTest(
            TestNetworkDesc(net_path, EXPERIMENTAL)
                .setUserInputPrecision("input", Precision::FP16)
                .setUserOutputPrecision("PostProcess/stage0/x1/Sigmoid", Precision::FP16)
                .setUserOutputPrecision("PostProcess/stage0/x4/Sigmoid", Precision::FP16)
                .setUserOutputPrecision("PostProcess/stage1/x1/Sigmoid", Precision::FP16)
                .setUserOutputPrecision("PostProcess/stage1/x4/Sigmoid", Precision::FP16)
                .setCompileConfig({{"VPU_COMPILER_NUM_CLUSTER", clusters}}),
                netName);

        std::cout << "Number of clusters: " << clusters << std::endl;
        std::cout << "Time elapsed: " << timeMs << std::endl;
    } catch (const import_error& ex) {
        std::cerr << ex.what() << std::endl;
        SKIP() << ex.what();
    }
}

static const std::vector<ClusterTestParams> params {
   ClusterTestParams("oneCluster", "1"),
   ClusterTestParams("fourClusters", "4"),
};

INSTANTIATE_TEST_CASE_P(perf, KmbClusterTestWithParams, testing::ValuesIn(params));
