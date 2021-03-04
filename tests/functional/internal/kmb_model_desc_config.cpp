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

#include <condition_variable>
#include "test_model/kmb_test_base.hpp"
#include "functional_test_utils/plugin_cache.hpp"

class KmbModelDescTest : public KmbNetworkTestBase {
public:
    int32_t runTest(const TestNetworkDesc& netDesc, const std::string& netFileName);
};

int32_t KmbModelDescTest::runTest(const TestNetworkDesc& netDesc, const std::string& netFileName) {
    const auto blobFileName = vpu::formatString("%v/%v.net", KmbTestBase::DUMP_PATH, netFileName);

#if defined(__arm__) || defined(__aarch64__)
    ExecutableNetwork exeNet = core->ImportNetwork(blobFileName, DEVICE_NAME, {});
#else
    CNNNetwork cnnNet = KmbNetworkTestBase::readNetwork(netDesc, true);
    ExecutableNetwork exeNet = core->LoadNetwork(cnnNet, DEVICE_NAME, netDesc.compileConfig());
    exeNet.Export(blobFileName);
#endif

    constexpr size_t MAX_ITER_COUNT = 1000;
    constexpr size_t MAX_REQ_COUNT = 8;
    std::condition_variable condVar;
    size_t iterCount = 0;

    if (KmbTestBase::RUN_INFER) {
        std::vector<InferenceEngine::InferRequest> inferRequestVec;
        for (size_t reqId = 0; reqId < MAX_REQ_COUNT; reqId++) {
            inferRequestVec.push_back(exeNet.CreateInferRequest());
            inferRequestVec.at(reqId).SetCompletionCallback(
                    [reqId, &inferRequestVec, &condVar, &iterCount, &MAX_ITER_COUNT] {
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
        condVar.wait(lock, [&] {
            return iterCount >= MAX_ITER_COUNT;
        });
        const auto msAfterRequest = std::chrono::steady_clock::now();

        for (auto& inferRequest : inferRequestVec) {
            inferRequest.Wait(-1);
        }

        const auto requestDurationSec =
                std::chrono::duration_cast<std::chrono::milliseconds>(msAfterRequest - msBeforeRequest);
        const auto duration = requestDurationSec.count();
        if (duration > 0) {
            return (MAX_ITER_COUNT / (double)duration) * 1000;
        }
    }

    return 0;
}

// [Track number: W#6518]
TEST_F(KmbModelDescTest, DISABLED_checkInferTimeWithAndWithoutConfig_ADK3) {
    const std::string net_path = "ADK3/ModelE_INT8/ModelE_INT8.xml";
    constexpr bool EXPERIMENTAL = true;

    const std::string withoutModelDescName = "withoutModelDesc";
    const auto withoutModelDescFPS =
            runTest(TestNetworkDesc(net_path, EXPERIMENTAL)
                            .setUserInputPrecision("input", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage0/x1/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage0/x4/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage1/x1/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage1/x4/Sigmoid", Precision::FP16)
                            .setCompileConfig({{"LOG_LEVEL", "LOG_DEBUG"}}),
                    withoutModelDescName);

    std::cout << "Without Model Description FPS: : " << withoutModelDescFPS << std::endl;
    
    const std::string withModelDescName = "withModelDesc";
    const auto withModelDescFPS =
            runTest(TestNetworkDesc(net_path, EXPERIMENTAL)
                            .setUserInputPrecision("input", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage0/x1/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage0/x4/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage1/x1/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage1/x4/Sigmoid", Precision::FP16)
                            .setCompileConfig({{"VPU_COMPILER_COMPILATION_DESCRIPTOR", "release_kmb_A0_1_cluster"}}),
                     withModelDescName);


    std::cout << "With Model Description FPS: " << withModelDescFPS << std::endl;
}

TEST_F(KmbModelDescTest, checkInferTime_ADK3) {
    const std::string net_path = "ADK3/ModelE_INT8/ModelE_INT8.xml";
    constexpr bool EXPERIMENTAL = true;

    const std::string withModelDescName = "withModelDesc";
    const auto withModelDescFPS =
            runTest(TestNetworkDesc(net_path, EXPERIMENTAL)
                            .setUserInputPrecision("input", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage0/x1/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage0/x4/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage1/x1/Sigmoid", Precision::FP16)
                            .setUserOutputPrecision("PostProcess/stage1/x4/Sigmoid", Precision::FP16)
                            .setCompileConfig({{"VPU_COMPILER_COMPILATION_DESCRIPTOR", "release_kmb_A0_1_cluster"}}),
                    withModelDescName);

    std::cout << "With Model Description FPS: " << withModelDescFPS << std::endl;
}
