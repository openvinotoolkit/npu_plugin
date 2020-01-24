//
// Copyright 2019 Intel Corporation.
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
#ifdef ENABLE_VPUAL

#include <file_reader.h>
#include <gtest/gtest.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <regression_tests.hpp>
#include <sstream>
#include <vpu_layers_tests.hpp>

#include "kmb_layers_tests.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace ::testing;

class KmbRegressionMultipleNetworks :
    public vpuLayersTests,
    public testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {};

TEST_P(KmbRegressionMultipleNetworks,
    canRunInferTwoNetworksSeveralIteration) {  // Hangs in Release time to time: CVS-23514
    auto param = GetParam();
    auto models = get<1>(param);

    InferenceEngine::ExecutableNetwork network1;
    stringstream network1StrStream;
    network1StrStream << "/KMB_models/BLOBS/" << get<0>(models) << "/" << get<0>(models) << ".blob";
    ASSERT_NO_THROW(network1 = ie.ImportNetwork(ModelsPath() + network1StrStream.str(), "KMB", {}));

    stringstream network2StrStream;
    network2StrStream << "/KMB_models/BLOBS/" << get<1>(models) << "/" << get<1>(models) << ".blob";
    InferenceEngine::ExecutableNetwork network2;
    ASSERT_NO_THROW(network2 = ie.ImportNetwork(ModelsPath() + network2StrStream.str(), "KMB", {}));

    std::cout << "Created networks\n";

    ASSERT_EQ(1, network1.GetInputsInfo().size());
    ASSERT_EQ(1, network2.GetInputsInfo().size());
    std::cout << "Input info is OK\n";

    auto createInferRequestAndWriteData = [](ExecutableNetwork& network,
                                              const std::string& filename) -> InferRequest::Ptr {
        InferenceEngine::InferRequest::Ptr inferRequestPtr;
        inferRequestPtr = network.CreateInferRequestPtr();

        auto inputInfo = network.GetInputsInfo();
        auto inputName = inputInfo.begin()->first;

        Blob::Ptr inputBlob;
        inputBlob = inferRequestPtr->GetBlob(inputName);
        vpu::KmbPlugin::utils::fromBinaryFile(filename, inputBlob);

        return inferRequestPtr;
    };

    stringstream input1StrStream;
    input1StrStream << "/KMB_models/BLOBS/" << get<0>(models) << "/input.bin";

    InferRequest::Ptr network1InferReqPtr;
    ASSERT_NO_THROW(
        network1InferReqPtr = createInferRequestAndWriteData(network1, ModelsPath() + input1StrStream.str()));

    stringstream input2StrStream;
    input2StrStream << "/KMB_models/BLOBS/" << get<1>(models) << "/input.bin";

    InferRequest::Ptr network2InferReqPtr;
    ASSERT_NO_THROW(
        network2InferReqPtr = createInferRequestAndWriteData(network2, ModelsPath() + input2StrStream.str()));

    std::cout << "Created inference requests\n";

    ASSERT_EQ(1, network1.GetOutputsInfo().size());
    ASSERT_EQ(1, network2.GetOutputsInfo().size());
    std::cout << "Output info is OK\n";

    const auto iterationCount = 5;
    std::string mode = get<0>(param);
    if (mode == "sync") {
        for (auto i = 0; i < iterationCount; i++) {
            ASSERT_NO_THROW(network1InferReqPtr->Infer());
            ASSERT_NO_THROW(network2InferReqPtr->Infer());
            std::cout << "Done " << i + 1 << " iteration of inference.\n";
        }
    } else if (mode == "async") {
        size_t curIterationNetwork1 = 0;
        size_t curIterationNet2 = 0;
        std::condition_variable condVar;

        network1InferReqPtr->SetCompletionCallback([&] {
            curIterationNetwork1++;
            std::cout << "Completed " << curIterationNetwork1 << " async request execution for network1\n";
            if (curIterationNetwork1 < static_cast<size_t>(iterationCount)) {
                Blob::Ptr outputBlob;
                std::string output1Name = network1.GetOutputsInfo().begin()->first;
                ASSERT_NO_THROW(outputBlob = network1InferReqPtr->GetBlob(output1Name));
                network1InferReqPtr->StartAsync();
            } else {
                condVar.notify_one();
            }
        });
        network2InferReqPtr->SetCompletionCallback([&] {
            curIterationNet2++;
            std::cout << "Completed " << curIterationNet2 << " async request execution for network1\n";
            if (curIterationNet2 < static_cast<size_t>(iterationCount)) {
                Blob::Ptr outputBlob;
                std::string output2Name = network2.GetOutputsInfo().begin()->first;
                ASSERT_NO_THROW(outputBlob = network2InferReqPtr->GetBlob(output2Name));
                network2InferReqPtr->StartAsync();
            } else {
                condVar.notify_one();
            }
        });

        std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network1" << std::endl;
        network1InferReqPtr->StartAsync();
        std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network2" << std::endl;
        network2InferReqPtr->StartAsync();

        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&] {
            return curIterationNetwork1 == static_cast<size_t>(iterationCount) &&
                   curIterationNet2 == static_cast<size_t>(iterationCount);
        });
    }
}

const static std::vector<std::string> executionMode = {"sync", "async"};

const static std::vector<std::pair<std::string, std::string>> modelPairs = {
    {"mobilenet-v2-dpu", "tiny-yolo-v2-dpu"}, {"mobilenet-v2-dpu", "resnet-50-dpu"}, {"resnet-50-dpu", "tiny-yolo-v2-dpu"}};

INSTANTIATE_TEST_CASE_P(
    inference, KmbRegressionMultipleNetworks, Combine(ValuesIn(executionMode), ValuesIn(modelPairs)));

#endif
