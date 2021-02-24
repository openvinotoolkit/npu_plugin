//
// Copyright 2020 Intel Corporation.
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
#include <mutex>
#include <string>

#include <vpu/utils/ie_helpers.hpp>

struct AsyncTestParams final {
    PARAMETER(std::size_t, nireq);
    PARAMETER(std::size_t, niter);
};

std::ostream& operator<<(std::ostream& os, const AsyncTestParams &p) {
    vpu::formatPrint(os, "nireq: %u, niter: %u", p.nireq(), p.niter());
    return os;
}

class VpuxAsyncTests: public KmbTestBase, public testing::WithParamInterface<AsyncTestParams> {};

TEST_P(VpuxAsyncTests, regression_ADK) {
    // for KMB/TBH standalone (VPUAL backend): there is a data race, the test sporadicaly fails: S#49626
    // TODO: it makes sense to introduce a separate macro to such SKIP
    #if defined(__arm__) || defined(__aarch64__)
    SKIP() << "Skip the test due to a data race on the current configuration";
    #endif
    // for HDDL by-pass (HDDL2 backend):  there is a data race, the test sporadicaly fails: S#49627
    SKIP_INFER_BYPASS_ON("VPUX", "data race");

    const auto &p = GetParam();
    const std::size_t nireq = p.nireq();
    const std::size_t niter = p.niter();

    if (RUN_COMPILER) {
        const auto precision = Precision::U8;
        const auto layout = Layout::NHWC;
        const std::vector<size_t> dims = {1, 1, 10, 10};

        const auto scaleDesc = TensorDesc(Precision::FP32, dims, layout);
        registerBlobGenerator(
                "scale", scaleDesc,
                [&](const TensorDesc& desc) {
                    return makeSingleValueBlob(desc, 1.f);
                }
        );

        const auto netPrecision = Precision::FP32;
        const auto inputDesc = TensorDesc(precision, dims, layout);
        TestNetwork testNet;
        testNet
            .setUserInput("input", inputDesc.getPrecision(), inputDesc.getLayout())
            .addNetInput("input", inputDesc.getDims(), netPrecision)
            .addLayer<PowerLayerDef>("power")
                .input1("input")
                .input2(getBlobByName("scale"))
                .build()
            .setUserOutput(PortInfo("power"), inputDesc.getPrecision(), inputDesc.getLayout())
            .addNetOutput(PortInfo("power"))
            .finalize();

        const auto cnnNet = testNet.getCNNNetwork();
        auto exeNet = core->LoadNetwork(cnnNet, DEVICE_NAME);

        KmbTestBase::exportNetwork(exeNet);
    }

    if (RUN_INFER) {
        auto exeNet = KmbTestBase::importNetwork();
        std::atomic<std::size_t> iterationCount(0);
        std::condition_variable waitCompletion;
        std::vector<InferenceEngine::InferRequest> inferRequests(nireq);
        std::vector<Blob::Ptr> inputs(nireq);
        std::atomic<std::size_t> idleReqs(0);
        const auto& inputDesc = exeNet.GetInputsInfo().begin()->second->getTensorDesc();
        for (std::size_t i = 0; i < nireq; i++) {
            registerBlobGenerator(
                    std::string("input") + std::to_string(i), inputDesc,
                    [&](const TensorDesc& desc) {
                        return makeSingleValueBlob(desc, 1.f + i);
                    }
            );
            inferRequests[i] = exeNet.CreateInferRequest();
            inputs[i] = KmbTestBase::getBlobByName(std::string("input") + std::to_string(i));
            inferRequests[i].SetBlob("input", inputs[i]);
            auto onComplete = [&waitCompletion, &iterationCount, &idleReqs, &inferRequests, i, &inputs, &niter, this](void) -> void {
                iterationCount++;
                if (iterationCount < niter) {
                    Blob::Ptr output = inferRequests[i].GetBlob("power");
                    KmbTestBase::compareOutputs(inputs[i], output, 0, CompareMethod::Absolute);
                    inferRequests[i].StartAsync();
                }
                else {
                    idleReqs++;
                    waitCompletion.notify_one();
                }
            };

            inferRequests[i].SetCompletionCallback(onComplete);
        }

        for (std::size_t i = 0; i < nireq; i++) {
            inferRequests[i].StartAsync();
        }

        std::mutex execGuard;
        std::unique_lock<std::mutex> execLocker(execGuard);
        waitCompletion.wait(execLocker, [&] {
            return idleReqs == nireq;
        });
    }
}

const std::vector<AsyncTestParams> asyncParams = {
        AsyncTestParams().nireq(1).niter(100),
        AsyncTestParams().nireq(2).niter(100),
        AsyncTestParams().nireq(4).niter(100),
        AsyncTestParams().nireq(8).niter(100),
};

INSTANTIATE_TEST_CASE_P(precommit, VpuxAsyncTests, testing::ValuesIn(asyncParams));
