//
// Copyright 2019-2020 Intel Corporation.
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

#include <creators/creator_blob_nv12.h>
#include <hddl2_helpers/helper_remote_blob.h>
#include <hddl2_helpers/helper_remote_memory.h>
#include <hddl2_helpers/helper_workload_context.h>
#include <helper_calc_cpu_ref.h>
#include <helper_remote_context.h>
#include <yolo_helpers.hpp>

#include <fstream>

#include "comparators.h"
#include "hddl2_load_network.h"
#include "ie_metric_helpers.hpp"
#include "tests_common.hpp"

#include "vpux/utils/IE/blob.hpp"

namespace IE = InferenceEngine;

class InferRequest_Tests : public ExecutableNetwork_Tests {
protected:
    void SetUp() override;
};

void InferRequest_Tests::SetUp() {
    ExecutableNetwork_Tests::SetUp();
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());
}

//------------------------------------------------------------------------------
TEST_F(InferRequest_Tests, CanCallInference) { ASSERT_NO_THROW(inferRequest.Infer()); }

//------------------------------------------------------------------------------
// TODO [Add tests] Set NV12Blob preprocessing information inside cnnNetwork
using InferRequest_SetBlob = InferRequest_Tests;

TEST_F(InferRequest_SetBlob, LocalBlob) {
    const std::string inputName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetworkPtr->GetInputsInfo().begin()->second;

    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    auto blob = IE::make_shared_blob<uint8_t>(inputTensorDesc);
    blob->allocate();

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, blob));
}

// TODO Simplify this test
TEST_F(InferRequest_SetBlob, RemoteBlob) {
    WorkloadContext_Helper workloadContextHelper;

    const std::string inputName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetworkPtr->GetInputsInfo().begin()->second;

    WorkloadID id = workloadContextHelper.getWorkloadId();
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(id);
    IE::RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);
    ASSERT_NE(remoteContext, nullptr);

    RemoteMemory_Helper remoteMemory;
    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    auto remoteMemoryFD = remoteMemory.allocateRemoteMemory(id, inputTensorDesc);
    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContext->CreateBlob(inputInfoPtr->getTensorDesc(), blobParams);
    ASSERT_NE(nullptr, remoteBlobPtr);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, remoteBlobPtr));
}

//------------------------------------------------------------------------------
class InferRequest_NV12 : public CoreAPI_Tests {
protected:
    void SetUp() override;
};

void InferRequest_NV12::SetUp() {
    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>(ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel));
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());
}

using InferRequest_NV12_SetBlob = InferRequest_NV12;
// TODO Long test
TEST_F(InferRequest_NV12_SetBlob, NV12Blob_WithPreprocessData_long) {
    ASSERT_EQ(executableNetworkPtr->GetInputsInfo().size(), 1);

    const std::string inputName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetworkPtr->GetInputsInfo().begin()->second;

    // TODO size shall be divided by two (nv12 calculation). Because of this use mobilenet
    auto nv12Blob = NV12Blob_Creator::createBlob(inputInfoPtr->getTensorDesc());
    auto preProcess = IE::PreProcessInfo();
    preProcess.setResizeAlgorithm(IE::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcess.setColorFormat(IE::ColorFormat::NV12);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, nv12Blob, preProcess));
}

//------------------------------------------------------------------------------
using InferRequest_GetBlob = InferRequest_Tests;

TEST_F(InferRequest_GetBlob, GetOutputAfterInference) {
    ASSERT_NO_THROW(inferRequest.Infer());

    std::string outputName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    IE::Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(outputName));
}

TEST_F(InferRequest_GetBlob, InputRemoteBlobContainSameDataAsOnSet) {
    WorkloadContext_Helper workloadContextHelper;
    const std::string inputName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetworkPtr->GetInputsInfo().begin()->second;

    WorkloadID id = workloadContextHelper.getWorkloadId();
    IE::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(id);
    IE::RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);
    ASSERT_NE(remoteContext, nullptr);

    RemoteMemory_Helper remoteMemoryHelper;
    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    auto remoteMemoryFD = remoteMemoryHelper.allocateRemoteMemory(id, inputTensorDesc);
    remoteMemoryHelper.clearRemoteMemory();
    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContext->CreateBlob(inputInfoPtr->getTensorDesc(), blobParams);
    ASSERT_NE(nullptr, remoteBlobPtr);

    const std::string inputData = "Test data";
    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto rBlobData = lockedMemory.as<char*>();
        memcpy(rBlobData, inputData.data(), inputData.size());
    }

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, remoteBlobPtr));

    std::string resultData;
    {
        IE::Blob::Ptr inputBlob = inferRequest.GetBlob(inputName);
        auto lockedMemory = inputBlob->cbuffer();
        auto rBlobData = lockedMemory.as<char*>();
        resultData = std::string(rBlobData);
    }

    ASSERT_EQ(inputData, resultData);
}

// TODO: unsetenv is not supported on windows platform
#ifdef __unix__

//------------------------------------------------------------------------------
using InferRequestCreation_Tests = CoreAPI_Tests;
// TODO Need to set env variable back after unset
TEST_F(InferRequestCreation_Tests, DISABLED_CanCompileButCanNotCreateRequestWithoutDaemon) {
    unsetenv("KMB_INSTALL_DIR");

    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>(ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel));
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());
}

#endif

//------------------------------------------------------------------------------
class Inference_onSpecificDevice : public CoreAPI_Tests {
public:
    int amountOfDevices = 0;

    const size_t numberOfTopClassesToCompare = 3;

protected:
    void SetUp() override;
};

void Inference_onSpecificDevice::SetUp() {
    const Models::ModelDesc modelToUse = Models::squeezenet1_1;
    network = ExecutableNetworkFactory::createCNNNetwork(modelToUse.pathToModel);

    std::vector<HddlUnite::Device> devices;
    getAvailableDevices(devices);
    amountOfDevices = devices.size();
}

TEST_F(Inference_onSpecificDevice, precommit_CanInferOnSpecificDeviceFromPluginMetrics) {
    std::vector<std::string> availableDevices = ie.GetMetric(pluginName, METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_TRUE(!availableDevices.empty());

    const std::string device_name = pluginName + "." + availableDevices[0];
    ASSERT_NO_THROW(executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>(ie.LoadNetwork(network, device_name)));
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}

TEST_F(Inference_onSpecificDevice, precommit_CanInferOnSpecificDeviceFromGetAllDevices) {
    if (amountOfDevices <= 1) {
        GTEST_SKIP() << "Not enough devices for test";
    }
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    ASSERT_TRUE(!availableDevices.empty());

    std::vector<std::string> VPUXDevices;
    std::copy_if(availableDevices.begin(), availableDevices.end(), std::back_inserter(VPUXDevices),
        [this](const std::string& deviceName) {
            return deviceName.find(pluginName) != std::string::npos;
        });

    ASSERT_TRUE(!VPUXDevices.empty());

    ASSERT_NO_THROW(executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>(ie.LoadNetwork(network, VPUXDevices[0])));
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}

//------------------------------------------------------------------------------
static void dumpPerformance(const std::map<std::string, IE::InferenceEngineProfileInfo>& perfMap) {
    std::vector<std::pair<std::string, IE::InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
    std::sort(perfVec.begin(), perfVec.end(),
        [=](const std::pair<std::string, IE::InferenceEngineProfileInfo>& pair1,
            const std::pair<std::string, IE::InferenceEngineProfileInfo>& pair2) -> bool {
            return pair1.second.execution_index < pair2.second.execution_index;
        });

    for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
        std::string name = it->first;
        IE::InferenceEngineProfileInfo info = it->second;
        if (info.status == IE::InferenceEngineProfileInfo::EXECUTED) {
            std::cout << "HDDL2 time: '" << name.c_str() << "' is " << info.realTime_uSec / 1000.f << " ms." << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
class InferenceWithPerfCount : public CoreAPI_Tests {
public:
    const size_t numberOfTopClassesToCompare = 3;
    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

protected:
    void SetUp() override;
};

void InferenceWithPerfCount::SetUp() {
    network = ExecutableNetworkFactory::createCNNNetwork(modelToUse.pathToModel);
    executableNetworkPtr = nullptr;
}

// C++ exception "InferAsync FAILED! return code:-4"
// [Track number: S#51440]
TEST_F(InferenceWithPerfCount, DISABLED_precommit_SyncInferenceWithPerfCount) {
    // ---- Load inference engine instance
    IE::Core ie;
    std::map<std::string, std::string> _config = {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}};

    ASSERT_NO_THROW(executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>(ie.LoadNetwork(network, pluginName, _config)));
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());

    auto outputBlobName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    auto perfCounts = inferRequest.GetPerformanceCounts();

    dumpPerformance(perfCounts);

    ASSERT_GT(perfCounts.size(), 0);
    auto totalTime = perfCounts.find("Total scoring time")->second;
    ASSERT_GT(totalTime.realTime_uSec, 0);
}

//------------------------------------------------------------------------------
class InferenceWithCheckLayout : public ExecutableNetwork_Tests {
public:
    const size_t numberOfTopClassesToCompare = 3;
    IE::Blob::Ptr inputBlob = nullptr;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

protected:
    void SetUp() override;
};

void InferenceWithCheckLayout::SetUp() {
    ExecutableNetwork_Tests::SetUp();
    const auto& inputInfo = executableNetworkPtr->GetInputsInfo().begin()->second;
    const auto& inputLayout = inputInfo->getTensorDesc().getLayout();
    const bool isBGR = true;
    inputBlob = loadImage("cat3.bmp", inputWidth, inputHeight, inputLayout, isBGR);
    inferRequest = executableNetworkPtr->CreateInferRequest();
}

TEST_F(InferenceWithCheckLayout, precommit_SyncInferenceAndCheckLayout) {
    const auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    inferRequest.SetBlob(inputBlobName, inputBlob);
    inferRequest.Infer();

    const auto& inputInfo = executableNetworkPtr->GetInputsInfo().begin()->second;
    const auto& networkInputLayout = inputInfo->getTensorDesc().getLayout();
    const auto& blobInputLayout = inferRequest.GetBlob(inputBlobName)->getTensorDesc().getLayout();
    ASSERT_EQ(blobInputLayout, networkInputLayout);
}

static IE::MemoryBlob::Ptr copyBlob(IE::MemoryBlob::Ptr in) {
    const auto out = IE::as<IE::MemoryBlob>(make_blob_with_precision(in->getTensorDesc()));
    out->allocate();

    const auto inMem = in->rmap();
    const auto outMem = out->wmap();

    const auto inPtr = inMem.as<const uint8_t*>();
    VPUX_THROW_UNLESS(inPtr != nullptr, "Blob was not allocated");

    const auto outPtr = outMem.as<uint8_t*>();
    VPUX_THROW_UNLESS(outPtr != nullptr, "Blob was not allocated");

    std::copy_n(inPtr, in->byteSize(), outPtr);

    return out;
}

TEST_F(InferenceWithCheckLayout, precommit_CheckInputsLayoutAfterTwoInferences) {
    const auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    const auto firstInputBlob = copyBlob(IE::as<IE::MemoryBlob>(inputBlob));
    inferRequest.SetBlob(inputBlobName, firstInputBlob);
    inferRequest.Infer();

    const auto outputBlobName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    const auto firstOutputBlob = copyBlob(IE::as<IE::MemoryBlob>(inferRequest.GetBlob(outputBlobName)));

    const auto secondInputBlob = copyBlob(IE::as<IE::MemoryBlob>(inputBlob));
    inferRequest.SetBlob(inputBlobName, secondInputBlob);
    inferRequest.Infer();

    const auto secondOutputBlob = copyBlob(IE::as<IE::MemoryBlob>(inferRequest.GetBlob(outputBlobName)));

    ASSERT_NO_THROW(
        Comparators::compareTopClasses(
                    vpux::toFP32(firstOutputBlob),
                    vpux::toFP32(secondOutputBlob),
                    numberOfTopClassesToCompare));
}

//------------------------------------------------------------------------------
const static std::vector<IE::Layout> inputLayoutVariants = {
    IE::Layout::NCHW,
    IE::Layout::NHWC
};

const static std::vector<IE::Layout> blobInputLayoutVariants = {
    IE::Layout::NCHW,
    IE::Layout::NHWC
};

//------------------------------------------------------------------------------
class InferenceCheckPortsNetwork :
    public CoreAPI_Tests,
    public testing::WithParamInterface<std::tuple<IE::Layout, IE::Layout, bool>> {
public:
    std::string modelPath;
    std::string inputPath;
    size_t inputWidth;
    size_t inputHeight;
    const size_t numberOfTopClassesToCompare = 3;
    const bool orderedClasses = false;
    const IE::Precision inputPrecision = IE::Precision::U8;

protected:
    void SetUp() override;
};

void InferenceCheckPortsNetwork::SetUp() {
    Models::ModelDesc modelToUse = Models::squeezenet1_1;
    modelPath = modelToUse.pathToModel;
    inputWidth = modelToUse.width;
    inputHeight = modelToUse.height;
    inputPath = "cat3.bmp";
}

TEST_P(InferenceCheckPortsNetwork, common) {
    const auto& testParam = GetParam();
    const auto inputLayout = std::get<0>(testParam);
    const auto blobInputLayout = std::get<1>(testParam);
    const auto importNetwork = std::get<2>(testParam);
    std::cout << "Parameters: input layout = " << inputLayout << " blob input layout = " << blobInputLayout <<
        " importNetwork = " << importNetwork << std::endl;

    // --- CNN Network and inputs
    std::cout << "Reading network..." << std::endl;
    ASSERT_NO_THROW(network = ExecutableNetworkFactory::createCNNNetwork(modelPath));
    IE::InputsDataMap input_info = network.getInputsInfo();
    for (auto& item : input_info) {
        auto input_data = item.second;
        input_data->setLayout(inputLayout);
        input_data->setPrecision(inputPrecision);
    }

    if (importNetwork) {
        const std::string exportName = "tmpfile";
        IE::ExecutableNetwork exportExecutableNetwork;
        std::cout << "Loading network..." << std::endl;
        ASSERT_NO_THROW(exportExecutableNetwork = ie.LoadNetwork(network, pluginName));
        ASSERT_NO_THROW(exportExecutableNetwork.Export(exportName));
        std::cout << "Importing network..." << std::endl;
        ASSERT_NO_THROW(executableNetworkPtr = std::make_shared<IE::ExecutableNetwork> (ie.ImportNetwork(exportName, pluginName)));
        ASSERT_TRUE(std::remove(exportName.c_str()) == 0);
    } else {
        std::cout << "Loading network..." << std::endl;
        ASSERT_NO_THROW(executableNetworkPtr = std::make_shared<IE::ExecutableNetwork> (ie.LoadNetwork(network, pluginName)));
    }

    // --- Infer request
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());

    // --- Input Blob
    auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = loadImage(inputPath, inputWidth, inputHeight, blobInputLayout, false));
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // --- Infer
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get result
    auto outputBlobName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    IE::Blob::Ptr outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refInputBlob = vpux::toLayout(IE::as<IE::MemoryBlob>(inputBlob), inputLayout);
    IE::Blob::Ptr refOutputBlob;
    ASSERT_NO_THROW(refOutputBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, refInputBlob));
    if (orderedClasses) {
        ASSERT_NO_THROW(
            Comparators::compareTopClasses(
                        vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                        vpux::toFP32(IE::as<IE::MemoryBlob>(refOutputBlob)),
                        numberOfTopClassesToCompare));
    } else {
        ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
                            vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                            vpux::toFP32(IE::as<IE::MemoryBlob>(refOutputBlob)),
                            numberOfTopClassesToCompare));
    }
}

 INSTANTIATE_TEST_SUITE_P(CheckPorts, InferenceCheckPortsNetwork, testing::Combine(testing::ValuesIn(inputLayoutVariants),
    testing::ValuesIn(blobInputLayoutVariants), testing::Bool()));

//------------------------------------------------------------------------------
class InferenceCheckPortsYoloV3Network :
    public CoreAPI_Tests,
    public testing::WithParamInterface<IE::Layout> {
public:
    std::string modelPath;
    std::string inputPath;

    size_t inputWidth;
    size_t inputHeight;
    const int yolov3_classes = 80;
    const int yolov3_coords = 4;
    const int yolov3_num = 3;
    const std::vector<float> yolov3_anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0,
        61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    const float yolov3_threshold = 0.6;
    const float boxTolerance = 0.4;
    const float probTolerance = 0.4;

protected:
    void SetUp() override;
};

void InferenceCheckPortsYoloV3Network::SetUp() {
    Models::ModelDesc modelToUse = Models::yolov3;
    modelPath = modelToUse.pathToModel;
    inputPath = "person.bmp";
    inputWidth = modelToUse.width;
    inputHeight = modelToUse.height;
}

TEST_P(InferenceCheckPortsYoloV3Network, common) {
    const auto blobInputLayout = GetParam();
    std::cout << "Parameters: blob input layout = " << blobInputLayout <<std::endl;

    std::cout << "Loading network..." << std::endl;
    ASSERT_NO_THROW(executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>
        (ExecutableNetworkFactory::createExecutableNetwork(modelPath)));

    // --- Infer request
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());

    // --- Input Blob
    auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = loadImage(inputPath, inputWidth, inputHeight, blobInputLayout, false));
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // --- Infer
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get result
    IE::BlobMap outputBlobs;
    IE::ConstOutputsDataMap output_info = executableNetworkPtr->GetOutputsInfo();
    for (const auto& output : output_info) {
        const auto outputBlobName = output.first;
        auto outputBlob = inferRequest.GetBlob(outputBlobName);
        outputBlobs[outputBlobName] = outputBlob;
    }

    // --- Reference Blob
    IE::Blob::Ptr refInputBlob = inferRequest.GetBlob(inputBlobName);
    IE::BlobMap refOutputBlobs;
    ASSERT_NO_THROW(refOutputBlobs = ReferenceHelper::CalcCpuReferenceMultipleOutput(modelPath, refInputBlob));

    // --- Parsing and comparing results
    const IE::Layout yolov3_layout = outputBlobs.begin()->second->getTensorDesc().getLayout();
    const IE::Layout ref_yolov3_layout = refOutputBlobs.begin()->second->getTensorDesc().getLayout();
    auto YoloV3Output = utils::parseYoloV3Output(outputBlobs, inputWidth, inputHeight, yolov3_classes,
        yolov3_coords, yolov3_num, yolov3_anchors, yolov3_threshold, yolov3_layout);
    auto refYoloV3Output = utils::parseYoloV3Output(refOutputBlobs, inputWidth, inputHeight, yolov3_classes,
        yolov3_coords, yolov3_num, yolov3_anchors, yolov3_threshold, ref_yolov3_layout);
    IE_Core_Helper::checkBBoxOutputs(YoloV3Output, refYoloV3Output, inputWidth, inputHeight, boxTolerance, probTolerance);
}

 INSTANTIATE_TEST_SUITE_P(CheckPorts, InferenceCheckPortsYoloV3Network, testing::ValuesIn(blobInputLayoutVariants));
