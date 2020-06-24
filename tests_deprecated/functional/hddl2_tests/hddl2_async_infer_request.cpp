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

namespace IE = InferenceEngine;

class AsyncInferRequest_Tests : public CoreAPI_Tests {
public:
    const int REQUEST_LIMIT = 10;
    const int MAX_WAIT = 60000;

    const int TOP_CLASSES_TO_COMPARE = 1;

    std::string refInputPath;
    std::string refOutputPath;

protected:
    modelBlobInfo _blobInfo = PrecompiledResNet_Helper::resnet50;

    void SetUp() override;

    std::vector<InferenceEngine::InferRequest> createRequests(const int& numberOfRequests);

    static void loadCatImageToBlobForRequests(
        const std::string& blobName, std::vector<InferenceEngine::InferRequest>& requests);

    IE::Blob::Ptr loadReferenceToBlob(
        const std::string& pathToReference, const IE::Precision& precision = IE::Precision::FP16);
};

void AsyncInferRequest_Tests::SetUp() {
    executableNetwork = ie.ImportNetwork(_blobInfo.graphPath, pluginName);
    refInputPath = _blobInfo.inputPath;
    refOutputPath = _blobInfo.outputPath;
}

std::vector<InferenceEngine::InferRequest> AsyncInferRequest_Tests::createRequests(const int& numberOfRequests) {
    std::vector<InferenceEngine::InferRequest> requests;
    for (int requestCount = 0; requestCount < numberOfRequests; requestCount++) {
        InferenceEngine::InferRequest inferRequest;
        inferRequest = executableNetwork.CreateInferRequest();
        requests.push_back(inferRequest);
    }
    return requests;
}

void AsyncInferRequest_Tests::loadCatImageToBlobForRequests(
    const std::string& blobName, std::vector<InferenceEngine::InferRequest>& requests) {
    for (auto currentRequest : requests) {
        IE::Blob::Ptr blobPtr;
        auto inputBlob = loadCatImage();
        currentRequest.SetBlob(blobName, inputBlob);
    }
}

IE::Blob::Ptr AsyncInferRequest_Tests::loadReferenceToBlob(
    const std::string& pathToReference, const IE::Precision& precision) {
    IE::ConstOutputsDataMap outputInfo = executableNetwork.GetOutputsInfo();
    if (outputInfo.size() != 1) {
        THROW_IE_EXCEPTION << "Only one output is supported";
    }
    auto outputTensorDesc = outputInfo.begin()->second->getTensorDesc();
    outputTensorDesc.setPrecision(precision);
    auto refBlob = make_blob_with_precision(outputTensorDesc);
    refBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(pathToReference, refBlob);
    return refBlob;
}

//------------------------------------------------------------------------------
// TODO Refactor create infer request for async inference correctly - use benchmark app approach
TEST_F(AsyncInferRequest_Tests, asyncIsFasterThenSync) {
    using Time = std::chrono::high_resolution_clock::time_point;
    Time (&Now)() = std::chrono::high_resolution_clock::now;

    Time start_sync;
    Time end_sync;
    {
        // --- Create requests
        std::vector<InferenceEngine::InferRequest> requests = createRequests(REQUEST_LIMIT);
        auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
        loadCatImageToBlobForRequests(inputBlobName, requests);

        // --- Sync execution
        start_sync = Now();
        for (InferenceEngine::InferRequest& currentRequest : requests) {
            ASSERT_NO_THROW(currentRequest.Infer());
        }
        end_sync = Now();
    }

    Time start_async;
    Time end_async;
    {
        // --- Create requests
        std::vector<InferenceEngine::InferRequest> requests = createRequests(REQUEST_LIMIT);
        auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
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
        for (InferenceEngine::InferRequest& currentRequest : requests) {
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

TEST_F(AsyncInferRequest_Tests, correctResultSameInput) {
    // --- Create requests
    std::vector<InferenceEngine::InferRequest> requests = createRequests(REQUEST_LIMIT);
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    loadCatImageToBlobForRequests(inputBlobName, requests);

    // --- Specify callback
    std::mutex requestCounterGuard;
    volatile int completedRequests = 0;
    auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
        std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
        completedRequests++;
    };

    // --- Asynchronous execution
    for (InferenceEngine::InferRequest& currentRequest : requests) {
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
    IE::Blob::Ptr refBlob = loadReferenceToBlob(refOutputPath);

    // --- Compare output with reference
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    for (auto currentRequest : requests) {
        IE::Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = currentRequest.GetBlob(outputBlobName));
        ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(refBlob), TOP_CLASSES_TO_COMPARE));
    }
}

//------------------------------------------------------------------------------
class AsyncInferRequest_DifferentInput : public AsyncInferRequest_Tests {
public:
    struct Reference {
        Reference(
            const std::string& inputReferencePath, const std::string& outputReferencePath, const bool& isNV12 = false)
            : inputReferencePath(inputReferencePath), outputReferencePath(outputReferencePath), isNV12(isNV12) {};
        std::string inputReferencePath;
        std::string outputReferencePath;
        bool isNV12;
    };

    std::vector<Reference> references;

protected:
    void SetUp() override;
};

void AsyncInferRequest_DifferentInput::SetUp() {
    executableNetwork = ie.ImportNetwork(_blobInfo.graphPath, pluginName);
    std::vector<Reference> availableReferences;

    availableReferences.emplace_back(Reference(_blobInfo.inputPath, _blobInfo.outputPath));
    availableReferences.emplace_back(Reference(_blobInfo.nv12Input, _blobInfo.nv12Output, true));

    const uint32_t seed = 666;
    static auto randEngine = std::default_random_engine(seed);
    const int referencesToUse = availableReferences.size();
    auto randGen = std::bind(std::uniform_int_distribution<>(0, referencesToUse - 1), randEngine);

    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        references.push_back(availableReferences.at(randGen()));
    }
}

//------------------------------------------------------------------------------
TEST_F(AsyncInferRequest_DifferentInput, correctResultShuffledNV12And) {
    // --- Create requests
    std::vector<InferenceEngine::InferRequest> requests = createRequests(REQUEST_LIMIT);
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;

    // --- Load random reference images
    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        if (references.at(i).isNV12) {
            // TODO Fix to follow same approach as hello nv12 classification sample
            const size_t image_width = 228;
            const size_t image_height = 228;
            IE::NV12Blob::Ptr nv12InputBlob =
                NV12Blob_Creator::createFromFile(references.at(i).inputReferencePath, image_width, image_height);

            // Since it 228x228 image on 224x224 network, resize preprocessing also required
            IE::PreProcessInfo preprocInfo = requests.at(i).GetPreProcess(inputBlobName);
            preprocInfo.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            preprocInfo.setColorFormat(IE::ColorFormat::NV12);

            // ---- Set NV12 blob with preprocessing information
            requests.at(i).SetBlob(inputBlobName, nv12InputBlob, preprocInfo);
        } else {
            auto inputBlob = loadCatImage();
            ASSERT_NO_THROW(requests.at(i).SetBlob(inputBlobName, inputBlob));
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
    for (InferenceEngine::InferRequest& currentRequest : requests) {
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
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        // --- Reference Blob
        IE::Blob::Ptr refBlob;
        if (references.at(i).isNV12) {
            refBlob = loadReferenceToBlob(references.at(i).outputReferencePath, IE::Precision::U8);
        } else {
            refBlob = loadReferenceToBlob(references.at(i).outputReferencePath);
        }

        IE::Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = requests.at(i).GetBlob(outputBlobName));
        ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(refBlob), TOP_CLASSES_TO_COMPARE));
    }
}
