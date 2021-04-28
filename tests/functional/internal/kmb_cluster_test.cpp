//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "test_model/kmb_test_base.hpp"
#include <condition_variable>

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
    if (KmbTestBase::PLATFORM == "VPU3400_A0") {
        // FIXME [Track number: E#10416]
        SKIP() << "MCM Compiler error: Failed to pass runtime simulation";
    }
    const auto& p = GetParam();
    const std::string net_path = "ADK3/ModelE_INT8/ModelE_INT8.xml";
    constexpr bool EXPERIMENTAL = true;

    const std::string netName = p._netName;
    const std::string clusters = p._numClusters;
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
}

static const std::vector<ClusterTestParams> params {
   ClusterTestParams("oneCluster", "1"),
   ClusterTestParams("fourClusters", "4"),
};

INSTANTIATE_TEST_CASE_P(perf, KmbClusterTestWithParams, testing::ValuesIn(params));
