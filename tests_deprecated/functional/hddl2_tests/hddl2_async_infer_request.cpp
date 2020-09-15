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

#include <creators/creator_blob_nv12.h>
#include <ie_compound_blob.h>

#include <blob_factory.hpp>
#include <mutex>
#include <random>
#include <thread>

#include "cases/core_api.h"
#include "comparators.h"
#include "file_reader.h"
#include "ie_utils.hpp"
#include "models/precompiled_resnet.h"
#include <helper_calc_cpu_ref.h>
#include <tests_common.hpp>

namespace IE = InferenceEngine;

class AsyncInferRequest_Tests : public CoreAPI_Tests {
public:
    const int REQUEST_LIMIT = 10;
    const int MAX_WAIT = 60000;

    std::string modelPath;

    const size_t inputWidth = 224;
    const size_t inputHeight = 224;
    const size_t numberOfTopClassesToCompare = 4;

protected:
    void SetUp() override;

    std::vector<IE::InferRequest> createRequests(const int& numberOfRequests);

    static void loadCatImageToBlobForRequests(
        const std::string& blobName, std::vector<IE::InferRequest>& requests);
};

void AsyncInferRequest_Tests::SetUp() {
    std::string graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
    modelPath = PrecompiledResNet_Helper::resnet50.modelPath;
    executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>(ie.ImportNetwork(graphPath, pluginName));
}

std::vector<IE::InferRequest> AsyncInferRequest_Tests::createRequests(const int& numberOfRequests) {
    std::vector<IE::InferRequest> requests;
    for (int requestCount = 0; requestCount < numberOfRequests; requestCount++) {
        IE::InferRequest inferRequest;
        inferRequest = executableNetworkPtr->CreateInferRequest();
        requests.push_back(inferRequest);
    }
    return requests;
}

void AsyncInferRequest_Tests::loadCatImageToBlobForRequests(
    const std::string& blobName, std::vector<IE::InferRequest>& requests) {
    for (auto currentRequest : requests) {
        IE::Blob::Ptr blobPtr;
        auto inputBlob = loadCatImage();
        currentRequest.SetBlob(blobName, inputBlob);
    }
}

//------------------------------------------------------------------------------
// TODO Refactor create infer request for async inference correctly - use benchmark app approach
TEST_F(AsyncInferRequest_Tests, precommit_asyncIsFasterThenSync) {
    using Time = std::chrono::high_resolution_clock::time_point;
    Time (&Now)() = std::chrono::high_resolution_clock::now;

    Time start_sync;
    Time end_sync;
    {
        // --- Create requests
        std::vector<IE::InferRequest> requests = createRequests(REQUEST_LIMIT);
        auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
        loadCatImageToBlobForRequests(inputBlobName, requests);

        // --- Sync execution
        start_sync = Now();
        for (IE::InferRequest& currentRequest : requests) {
            ASSERT_NO_THROW(currentRequest.Infer());
        }
        end_sync = Now();
    }

    Time start_async;
    Time end_async;
    {
        // --- Create requests
        std::vector<IE::InferRequest> requests = createRequests(REQUEST_LIMIT);
        auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
        loadCatImageToBlobForRequests(inputBlobName, requests);

        // --- Specify callback
        std::mutex requestCounterGuard;
        volatile int completedRequests = 0;
        auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
            std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
            completedRequests++;
        };

        start_async = Now();
        // --- Asynchronous execution
        for (IE::InferRequest& currentRequest : requests) {
            currentRequest.SetCompletionCallback(onComplete);
            currentRequest.StartAsync();
        }

        auto waitRoutine = [&completedRequests, this]() -> void {
            std::chrono::system_clock::time_point endTime =
                std::chrono::system_clock::now() + std::chrono::milliseconds(MAX_WAIT);
            while (completedRequests < REQUEST_LIMIT) {
                ASSERT_LE(std::chrono::system_clock::now(), endTime);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };

        std::thread waitThread(waitRoutine);
        waitThread.join();
        end_async = Now();
    }

    auto elapsed_sync = std::chrono::duration_cast<std::chrono::milliseconds>(end_sync - start_sync);
    std::cout << "Sync inference (ms)" << elapsed_sync.count() << std::endl;

    auto elapsed_async = std::chrono::duration_cast<std::chrono::milliseconds>(end_async - start_async);
    std::cout << "Async inference (ms)" << elapsed_async.count() << std::endl;

    ASSERT_LT(elapsed_async.count(), elapsed_sync.count());
}

TEST_F(AsyncInferRequest_Tests, precommit_correctResultSameInput) {
    // --- Create requests
    std::vector<IE::InferRequest> requests = createRequests(REQUEST_LIMIT);
    auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    loadCatImageToBlobForRequests(inputBlobName, requests);

    // --- Specify callback
    std::mutex requestCounterGuard;
    volatile int completedRequests = 0;
    auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
        std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
        completedRequests++;
    };

    // --- Asynchronous execution
    for (IE::InferRequest& currentRequest : requests) {
        currentRequest.SetCompletionCallback(onComplete);
        ASSERT_NO_THROW(currentRequest.StartAsync());
    }

    auto waitRoutine = [&completedRequests, this]() -> void {
        std::chrono::system_clock::time_point endTime =
            std::chrono::system_clock::now() + std::chrono::milliseconds(MAX_WAIT);
        while (completedRequests < REQUEST_LIMIT) {
            ASSERT_LE(std::chrono::system_clock::now(), endTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    std::thread waitThread(waitRoutine);
    waitThread.join();

    // --- Reference Blob
    IE::Blob::Ptr inputBlob = requests.at(0).GetBlob(inputBlobName);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputBlob);

    // --- Compare output with reference
    auto outputBlobName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    for (auto currentRequest : requests) {
        IE::Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = currentRequest.GetBlob(outputBlobName));
        ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
    }
}

//------------------------------------------------------------------------------
class AsyncInferRequest_DifferentInput : public AsyncInferRequest_Tests {
public:
    struct Reference {
        explicit Reference(const bool _isNV12 = false)
            : isNV12(_isNV12) {}
        bool isNV12;
    };

    std::vector<Reference> references;
    std::string inputNV12Path;

protected:
    void SetUp() override;
};

void AsyncInferRequest_DifferentInput::SetUp() {
    AsyncInferRequest_Tests::SetUp();
    inputNV12Path = TestDataHelpers::get_data_path() + "/" + std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "/cat3.yuv";
    std::vector<Reference> availableReferences;

    availableReferences.emplace_back(Reference(false));
    availableReferences.emplace_back(Reference(true));

    const uint32_t seed = 666;
    static auto randEngine = std::default_random_engine(seed);
    const int referencesToUse = availableReferences.size();
    auto randGen = std::bind(std::uniform_int_distribution<>(0, referencesToUse - 1), randEngine);

    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        references.push_back(availableReferences.at(randGen()));
    }
}

//------------------------------------------------------------------------------
TEST_F(AsyncInferRequest_DifferentInput, precommit_correctResultShuffled_NoPreprocAndPreproc) {
    // --- Create requests
    std::vector<IE::InferRequest> requests = createRequests(REQUEST_LIMIT);
    auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::Blob::Ptr refRgbBlob = nullptr;
    IE::Blob::Ptr refNV12Blob = nullptr;

    // --- Load random reference images
    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        if (references.at(i).isNV12) {
            // TODO Fix to follow same approach as hello nv12 classification sample
            // ----- Load NV12 input
            IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(
                inputNV12Path, inputWidth, inputHeight);

            // Preprocessing
            IE::PreProcessInfo preprocInfo = requests.at(i).GetPreProcess(inputBlobName);
            preprocInfo.setColorFormat(IE::ColorFormat::NV12);

            // ---- Set NV12 blob with preprocessing information
            requests.at(i).SetBlob(inputBlobName, nv12InputBlob, preprocInfo);

            if (refNV12Blob == nullptr) {
                refNV12Blob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, nv12InputBlob, &preprocInfo);
            }
        } else {
            auto inputBlob = loadCatImage();
            ASSERT_NO_THROW(requests.at(i).SetBlob(inputBlobName, inputBlob));

            if (refRgbBlob == nullptr) {
                refRgbBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputBlob);
            }
        }
    }

    // --- Specify callback
    std::mutex requestCounterGuard;
    volatile int completedRequests = 0;
    auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
        std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
        completedRequests++;
    };

    // --- Asynchronous execution
    for (IE::InferRequest& currentRequest : requests) {
        currentRequest.SetCompletionCallback(onComplete);
        ASSERT_NO_THROW(currentRequest.StartAsync());
    }

    auto waitRoutine = [&completedRequests, this]() -> void {
        std::chrono::system_clock::time_point endTime =
            std::chrono::system_clock::now() + std::chrono::milliseconds(MAX_WAIT);
        while (completedRequests < REQUEST_LIMIT) {
            ASSERT_LE(std::chrono::system_clock::now(), endTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    std::thread waitThread(waitRoutine);
    waitThread.join();

    // --- Compare output with reference
    auto outputBlobName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    IE::Blob::Ptr refBlob;
    IE::Blob::Ptr outputBlob;
    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        refBlob = references.at(i).isNV12 ? refNV12Blob : refRgbBlob;
        ASSERT_NO_THROW(outputBlob = requests.at(i).GetBlob(outputBlobName));
        ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
    }
}
