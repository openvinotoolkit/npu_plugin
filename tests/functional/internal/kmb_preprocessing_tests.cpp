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

#if defined(__aarch64__)

#include <fcntl.h>
#include <file_reader.h>
#include <gtest/gtest.h>
#include <ie_compound_blob.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vpusmm/vpusmm.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vpux/vpux_plugin_config.hpp>
#include <test_model/kmb_test_base.hpp>

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

static std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> buildAllocator(const char* allocatorType) {
    if (allocatorType == nullptr) {
        return std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    }

    std::string allocTypeStr(allocatorType);
    if (allocTypeStr == "NATIVE") {
        return std::make_shared<vpu::KmbPlugin::utils::NativeAllocator>();
    } else if (allocTypeStr == "UDMA") {
        throw std::runtime_error("buildAllocator: UDMA is not supported");
    }

    // VPUSMM is default
    return std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
}

static void setNV12Preproc(const std::string& inputName, const std::string& inputFilePath,
    InferenceEngine::InferRequest& inferRequest, std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>& allocator,
    size_t expectedWidth, size_t expectedHeight) {
    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(
        inputBlob = vpu::KmbPlugin::utils::fromNV12File(inputFilePath, expectedWidth, expectedHeight, allocator));

    PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
    preprocInfo.setColorFormat(ColorFormat::NV12);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, inputBlob, preprocInfo));
}

// TODO consider re-using twoNetworksWithPreprocessing instead of duplicating it
using VpuPreprocessingStressTests = KmbYoloV2NetworkTest;

// [Track number: S#35173, S#35231]
TEST_F(VpuPreprocessingStressTests, DISABLED_twoNetworksHDImage1000Iterations) {
    if (!KmbTestBase::RUN_INFER) {
        SKIP();
    }
    Core ie;
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = getTestModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/schema-3.24.3/mobilenet-v2.blob";
    ASSERT_NO_THROW(network1 = ie.ImportNetwork(network1Path, DEVICE_NAME, {}));

    std::string network2Path = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/schema-3.24.3/tiny-yolo-v2.blob";
    InferenceEngine::ExecutableNetwork network2;
    ASSERT_NO_THROW(network2 = ie.ImportNetwork(network2Path, DEVICE_NAME, {}));

    std::cout << "Created networks\n";

    ASSERT_EQ(1, network1.GetInputsInfo().size());
    ASSERT_EQ(1, network2.GetInputsInfo().size());
    std::cout << "Input info is OK\n";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    InferenceEngine::InferRequest::Ptr network1InferReqPtr;
    network1InferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    ConstInputsDataMap inputInfo2 = network2.GetInputsInfo();

    const size_t imgWidth = 1920;
    const size_t imgHeight = 1080;

    std::string input1_name = inputInfo1.begin()->first;
    std::string input1Path = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-1-1920x1080-nv12.bin";
    setNV12Preproc(input1_name, input1Path, *network1InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    InferenceEngine::InferRequest::Ptr network2InferReqPtr;
    network2InferReqPtr = network2.CreateInferRequestPtr();

    std::string input2_name = inputInfo2.begin()->first;
    std::string input2Path = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-2-1920x1080-nv12.bin";
    setNV12Preproc(input2_name, input2Path, *network2InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    std::cout << "Created inference requests\n";

    ASSERT_EQ(1, network1.GetOutputsInfo().size());
    ASSERT_EQ(1, network2.GetOutputsInfo().size());
    std::cout << "Output info is OK\n";

    const auto iterationCount = 1000;
    size_t curIterationNetwork1 = 0;
    size_t curIterationNetwork2 = 0;
    std::condition_variable condVar;

    network1InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork1++;
        std::cout << "Completed " << curIterationNetwork1 << " async request execution for network 1" << std::endl;
        if (curIterationNetwork1 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            std::string output1Name = network1.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network1InferReqPtr->GetBlob(output1Name));
            network1InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });
    const size_t BBOX_PRINT_INTERVAL = 100;
    network2InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork2++;
        std::cout << "Completed " << curIterationNetwork2 << " async request execution for network 2" << std::endl;
        if (curIterationNetwork2 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            // throws [REQUEST_BUSY] when USE_SIPP=0
            // [Track number: S#35231]
            try {
                std::string output2Name = network2.GetOutputsInfo().begin()->first;
                ASSERT_NO_THROW(outputBlob = network2InferReqPtr->GetBlob(output2Name));

                if (curIterationNetwork2 % BBOX_PRINT_INTERVAL == 0) {
                    float confThresh = 0.4f;
                    bool isTiny = true;
                    auto actualOutput = utils::parseYoloOutput(outputBlob, imgWidth, imgHeight, confThresh, isTiny);
                    std::cout << "BBox Top:" << std::endl;
                    for (size_t i = 0; i < actualOutput.size(); ++i) {
                        const auto& bb = actualOutput[i];
                        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right
                                  << " " << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
                    }
                }
            } catch (const std::exception& exc) {
                std::cout << "detectCallback caught exception " << exc.what() << std::endl;
            }
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
               curIterationNetwork2 == static_cast<size_t>(iterationCount);
    });
}

// [Track number: S#35173, S#35231]
TEST_F(VpuPreprocessingStressTests, DISABLED_twoNetworksStressTest) {
    if (!KmbTestBase::RUN_INFER) {
        SKIP();
    }
    Core ie;
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = getTestModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/schema-3.24.3/mobilenet-v2.blob";
    ASSERT_NO_THROW(network1 = ie.ImportNetwork(network1Path, DEVICE_NAME, {}));

    std::string network2Path = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/schema-3.24.3/tiny-yolo-v2.blob";
    InferenceEngine::ExecutableNetwork network2;
    ASSERT_NO_THROW(network2 = ie.ImportNetwork(network2Path, DEVICE_NAME, {}));

    std::cout << "Created networks\n";

    ASSERT_EQ(1, network1.GetInputsInfo().size());
    ASSERT_EQ(1, network2.GetInputsInfo().size());
    std::cout << "Input info is OK\n";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    InferenceEngine::InferRequest::Ptr network1InferReqPtr;
    network1InferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    ConstInputsDataMap inputInfo2 = network2.GetInputsInfo();

    const size_t imgWidth = 1920;
    const size_t imgHeight = 1080;

    std::string input1_name = inputInfo1.begin()->first;
    std::string input1Path = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-1-1920x1080-nv12.bin";
    setNV12Preproc(input1_name, input1Path, *network1InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    InferenceEngine::InferRequest::Ptr network2InferReqPtr;
    network2InferReqPtr = network2.CreateInferRequestPtr();

    std::string input2_name = inputInfo2.begin()->first;
    std::string input2Path = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-2-1920x1080-nv12.bin";
    setNV12Preproc(input2_name, input2Path, *network2InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    std::cout << "Created inference requests\n";

    ASSERT_EQ(1, network1.GetOutputsInfo().size());
    ASSERT_EQ(1, network2.GetOutputsInfo().size());
    std::cout << "Output info is OK\n";

    const std::chrono::system_clock::time_point timeLimit =
        std::chrono::system_clock::now() + std::chrono::seconds(10 * 60);  // 10 minutes
    size_t curIterationNetwork1 = 0;
    size_t curIterationNetwork2 = 0;
    std::atomic<bool> network1Finished(false);
    std::atomic<bool> network2Finished(false);
    std::atomic<bool> network1Failed(false);
    std::atomic<bool> network2Failed(false);
    std::condition_variable condVar;

    network1InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork1++;
        std::cout << "Completed " << curIterationNetwork1 << " async request execution for network 1" << std::endl;
        if (std::chrono::system_clock::now() < timeLimit) {
            try {
                Blob::Ptr outputBlob;
                std::string output1Name = network1.GetOutputsInfo().begin()->first;
                ASSERT_NO_THROW(outputBlob = network1InferReqPtr->GetBlob(output1Name));
                network1InferReqPtr->StartAsync();
            } catch (const std::exception& exc) {
                std::cout << "classifyCallback caught exception " << exc.what() << std::endl;
                network1Finished = true;
                network1Failed = true;
                condVar.notify_one();
            }
        } else {
            network1Finished = true;
            network1Failed = false;
            condVar.notify_one();
        }
    });

    const size_t BBOX_PRINT_INTERVAL = 100;
    network2InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork2++;
        std::cout << "Completed " << curIterationNetwork2 << " async request execution for network 2" << std::endl;
        if (std::chrono::system_clock::now() < timeLimit) {
            // throws [REQUEST_BUSY] when USE_SIPP=0
            // [Track number: S#35231]
            try {
                Blob::Ptr outputBlob;
                std::string output2Name = network2.GetOutputsInfo().begin()->first;
                ASSERT_NO_THROW(outputBlob = network2InferReqPtr->GetBlob(output2Name));

                if (curIterationNetwork2 % BBOX_PRINT_INTERVAL == 0) {
                    float confThresh = 0.4f;
                    bool isTiny = true;
                    auto actualOutput = utils::parseYoloOutput(outputBlob, imgWidth, imgHeight, confThresh, isTiny);
                    std::cout << "BBox Top:" << std::endl;
                    for (size_t i = 0; i < actualOutput.size(); ++i) {
                        const auto& bb = actualOutput[i];
                        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right
                                  << " " << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
                    }
                }
                network2InferReqPtr->StartAsync();
            } catch (const std::exception& exc) {
                std::cout << "detectCallback caught exception " << exc.what() << std::endl;
                network2Finished = true;
                network2Failed = true;
                condVar.notify_one();
            }
        } else {
            network2Finished = true;
            network2Failed = false;
            condVar.notify_one();
        }
    });

    std::cout << "Start inference for network1" << std::endl;
    network1InferReqPtr->StartAsync();
    std::cout << "Start inference for network2" << std::endl;
    network2InferReqPtr->StartAsync();

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [&] {
        return network1Finished && network2Finished;
    });

    ASSERT_FALSE(network1Failed);
    ASSERT_FALSE(network2Failed);
}

// [Track number: S#35173, S#35231]
TEST_F(VpuPreprocessingStressTests, DISABLED_detectClassify4Threads) {
    if (!KmbTestBase::RUN_INFER) {
        SKIP();
    }
    Core ie;

    std::string detectNetworkPath = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/schema-3.24.3/tiny-yolo-v2.blob";
    InferenceEngine::ExecutableNetwork detectionNetwork = ie.ImportNetwork(detectNetworkPath, DEVICE_NAME, {});

    std::string classifyNetworkPath = getTestModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/schema-3.24.3/mobilenet-v2.blob";
    InferenceEngine::ExecutableNetwork classificationNetwork = ie.ImportNetwork(classifyNetworkPath, DEVICE_NAME, {});

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    std::vector<InferenceEngine::InferRequest::Ptr> detectionRequests;
    std::vector<InferenceEngine::InferRequest::Ptr> classificationRequests;
    ConstInputsDataMap detectInputInfo = detectionNetwork.GetInputsInfo();
    ConstInputsDataMap classifyInputInfo = classificationNetwork.GetInputsInfo();

    const size_t imgWidth = 1920;
    const size_t imgHeight = 1080;
    const size_t maxParallelRequests = 4;

    std::condition_variable condVar;
    const char* iterCountStr = std::getenv("IE_VPU_KMB_STRESS_TEST_ITER_COUNT");
    const size_t iterationCount = (iterCountStr != nullptr) ? std::atoi(iterCountStr) : 10;
    std::vector<size_t> iterationVec(maxParallelRequests, 0);

    std::string detectInputName = detectInputInfo.begin()->first;
    std::string detectInputPath = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-1-1920x1080-nv12.bin";
    std::string classifyInputName = classifyInputInfo.begin()->first;
    std::string classifyInputPath = getTestModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-2-1920x1080-nv12.bin";

    std::string detectOutputName = detectionNetwork.GetOutputsInfo().begin()->first;
    std::string classifyOutputName = classificationNetwork.GetOutputsInfo().begin()->first;

    for (size_t requestNum = 0; requestNum < maxParallelRequests; requestNum++) {
        InferenceEngine::InferRequest::Ptr detectInferReq = detectionNetwork.CreateInferRequestPtr();
        setNV12Preproc(detectInputName, detectInputPath, *detectInferReq, kmbAllocator, imgWidth, imgHeight);

        InferenceEngine::InferRequest::Ptr classifyInferReq = classificationNetwork.CreateInferRequestPtr();
        setNV12Preproc(classifyInputName, classifyInputPath, *classifyInferReq, kmbAllocator, imgWidth, imgHeight);

        auto detectCallback = [requestNum, &iterationVec, iterationCount, &detectionRequests, &classificationRequests,
                                  detectOutputName, &condVar, maxParallelRequests](void) -> void {
            size_t reqId = requestNum;
            iterationVec[reqId]++;
            std::cout << "Completed " << iterationVec[reqId] << " async request ID " << reqId << std::endl;
            if (iterationVec[reqId] < iterationCount) {
                // throws [REQUEST_BUSY] when USE_SIPP=0
                // [Track number: S#35231]
                try {
                    Blob::Ptr detectOutputBlob = detectionRequests.at(reqId)->GetBlob(detectOutputName);

                    float confThresh = 0.4f;
                    bool isTiny = true;
                    auto actualOutput =
                        utils::parseYoloOutput(detectOutputBlob, imgWidth, imgHeight, confThresh, isTiny);
                    std::cout << "BBox Top:" << std::endl;
                    for (size_t i = 0; i < actualOutput.size(); ++i) {
                        const auto& bb = actualOutput[i];
                        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right
                                  << " " << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
                    }
                    for (size_t classReqIdx = 0; classReqIdx < maxParallelRequests; classReqIdx++) {
                        classificationRequests.at(classReqIdx)->Wait(IInferRequest::WaitMode::RESULT_READY);
                        classificationRequests.at(classReqIdx)->StartAsync();
                    }
                } catch (const std::exception& exc) {
                    std::cout << "detectCallback caught exception " << exc.what() << std::endl;
                }
                detectionRequests.at(reqId)->StartAsync();
            } else {
                condVar.notify_one();
            }
        };

        auto classifyCallback = [requestNum, &classificationRequests, classifyOutputName](void) -> void {
            size_t reqId = requestNum;
            Blob::Ptr classifyOutputBlob = classificationRequests.at(reqId)->GetBlob(classifyOutputName);
            const float* bufferRawPtr = classifyOutputBlob->cbuffer().as<const float*>();
            std::vector<float> outputData(classifyOutputBlob->size());
            std::memcpy(outputData.data(), bufferRawPtr, outputData.size());
            std::vector<float>::iterator maxElt = std::max_element(outputData.begin(), outputData.end());
            std::cout << "Top class: " << std::distance(outputData.begin(), maxElt) << std::endl;
        };
        detectInferReq->SetCompletionCallback(detectCallback);
        detectionRequests.push_back(detectInferReq);
        classifyInferReq->SetCompletionCallback(classifyCallback);
        classificationRequests.push_back(classifyInferReq);
    }

    for (const auto& detectReq : detectionRequests) {
        detectReq->StartAsync();
    }

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [&] {
        bool allRequestsFinished = true;
        for (size_t iterNum : iterationVec) {
            if (iterNum < iterationCount) {
                allRequestsFinished = false;
            }
        }
        return allRequestsFinished;
    });
}

#endif
