//
// Copyright 2019-2020 Intel Corporation.
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

#include <executable_network.h>
#include <hddl2_helpers/helper_remote_blob.h>
#include <hddl2_helpers/helper_remote_memory.h>
#include <hddl2_helpers/helper_workload_context.h>
#include <helper_calc_cpu_ref.h>
#include <helper_remote_context.h>
#include <models/precompiled_resnet.h>

#include <blob_factory.hpp>
#include <chrono>
#include <fstream>
#include <ie_core.hpp>
#include <thread>
#include <vpu/utils/ie_helpers.hpp>

#include "comparators.h"
#include "creators/creator_blob_nv12.h"
#include "ie_metric_helpers.hpp"
#include "ie_utils.hpp"
#include "models/model_pooling.h"

namespace IE = InferenceEngine;

// TODO Use ImportNetwork tests as base
class InferRequest_Tests : public CoreAPI_Tests {
public:
    modelBlobInfo blobInfo = PrecompiledResNet_Helper::resnet50;

protected:
    void SetUp() override;
};

void InferRequest_Tests::SetUp() {
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));
}

//------------------------------------------------------------------------------
TEST_F(InferRequest_Tests, CanCreateInferRequest) {
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(InferRequest_Tests, CanCallInference) {
    inferRequest = executableNetwork.CreateInferRequest();

    ASSERT_NO_THROW(inferRequest.Infer());
}

//------------------------------------------------------------------------------
// TODO [Add tests] Set NV12Blob preprocessing information inside cnnNetwork
using InferRequest_SetBlob = InferRequest_Tests;
TEST_F(InferRequest_SetBlob, CanSetInputBlob) {
    inferRequest = executableNetwork.CreateInferRequest();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(inputTensorDesc);
    blob->allocate();

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, blob));
}

TEST_F(InferRequest_SetBlob, CanSetInput_RemoteBlob) {
    WorkloadContext_Helper workloadContextHelper;
    inferRequest = executableNetwork.CreateInferRequest();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    WorkloadID id = workloadContextHelper.getWorkloadId();
    InferenceEngine::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(id);
    IE::RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);
    ASSERT_NE(remoteContext, nullptr);

    RemoteMemory_Helper remoteMemory;
    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    RemoteMemoryFd memoryFd = remoteMemory.allocateRemoteMemory(id, inputTensorDesc);
    auto blobParams = RemoteBlob_Helper::wrapRemoteFdToMap(memoryFd);
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContext->CreateBlob(inputInfoPtr->getTensorDesc(), blobParams);
    ASSERT_NE(nullptr, remoteBlobPtr);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, remoteBlobPtr));
}

// [Track number: S#30141]
TEST_F(InferRequest_SetBlob, CanSetInput_NV12Blob_WithPreprocessData) {
    inferRequest = executableNetwork.CreateInferRequest();
    ASSERT_EQ(executableNetwork.GetInputsInfo().size(), 1);

    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    auto nv12Blob = NV12Blob_Creator::createBlob(inputInfoPtr->getTensorDesc());
    auto preProcess = IE::PreProcessInfo();
    preProcess.setResizeAlgorithm(IE::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcess.setColorFormat(IE::ColorFormat::NV12);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, nv12Blob, preProcess));
}

//------------------------------------------------------------------------------
using InferRequest_GetBlob = InferRequest_Tests;
TEST_F(InferRequest_GetBlob, CanGetOutputBlobAfterInference) {
    inferRequest = executableNetwork.CreateInferRequest();

    inferRequest.Infer();

    std::string outputName = executableNetwork.GetOutputsInfo().begin()->first;
    InferenceEngine::Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(outputName));
}

TEST_F(InferRequest_GetBlob, GetBlobWillContainsSameDataAsSetBlob_WithRemoteMemory) {
    WorkloadContext_Helper workloadContextHelper;
    inferRequest = executableNetwork.CreateInferRequest();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    WorkloadID id = workloadContextHelper.getWorkloadId();
    InferenceEngine::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(id);
    IE::RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);
    ASSERT_NE(remoteContext, nullptr);

    RemoteMemory_Helper remoteMemory;
    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    RemoteMemoryFd memoryFd = remoteMemory.allocateRemoteMemory(id, inputTensorDesc);
    auto blobParams = RemoteBlob_Helper::wrapRemoteFdToMap(memoryFd);
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
        auto inputBlobData = inputBlob->buffer().as<char*>();
        resultData = std::string(inputBlobData);
    }

    ASSERT_EQ(inputData, resultData);
}

//------------------------------------------------------------------------------
using InferRequestCreation_Tests = InferRequest_Tests;
TEST_F(InferRequestCreation_Tests, CanCompileButCanNotCreateRequestWithoutDaemon) {
    std::vector<std::string> devices = ie.GetMetric(pluginName, METRIC_KEY(AVAILABLE_DEVICES));
    if (!devices.empty()) {
        GTEST_SKIP() << "Not possible to test it with device / service.";
    }
    unsetenv("KMB_INSTALL_DIR");

    ASSERT_ANY_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

//------------------------------------------------------------------------------
class Inference_onSpecificDevice : public InferRequest_Tests {
public:
    int amountOfDevices = 0;

    std::string graphPath;
    std::string refInputPath;
    std::string refOutputPath;

    const size_t numberOfTopClassesToCompare = 5;

protected:
    void SetUp() override;
};

void Inference_onSpecificDevice::SetUp() {
    std::vector<HddlUnite::Device> devices;
    getAvailableDevices(devices);
    amountOfDevices = devices.size();
    graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50.inputPath;
    refOutputPath = PrecompiledResNet_Helper::resnet50.outputPath;
}

TEST_F(Inference_onSpecificDevice, precommit_CanInferOnSpecificDeviceFromPluginMetrics) {
    std::vector<std::string> availableDevices = ie.GetMetric(pluginName, METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_TRUE(!availableDevices.empty());

    const std::string device_name = pluginName + "." + availableDevices[0];
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}

TEST_F(Inference_onSpecificDevice, precommit_CanInferOnSpecificDeviceFromGetAllDevices) {
    if (amountOfDevices <= 1) {
        GTEST_SKIP() << "Not enough devices for test";
    }
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    ASSERT_TRUE(!availableDevices.empty());

    std::vector<std::string> HDDL2Devices;
    std::copy_if(availableDevices.begin(), availableDevices.end(), std::back_inserter(HDDL2Devices),
        [this](const std::string& deviceName) {
            return deviceName.find(pluginName) != std::string::npos;
        });

    ASSERT_TRUE(!HDDL2Devices.empty());

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, HDDL2Devices[0]));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}

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
            printf("HDDL2 time: '%s' is %f ms.\n", name.c_str(), info.realTime_uSec / 1000.f);
        }
    }
}

using InferenceWithPerfCount = Inference_onSpecificDevice;

TEST_F(InferenceWithPerfCount, precommit_SyncInferenceWithPerfCount) {
    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    std::map<std::string, std::string> _config = {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}};

    // ---- Import or load network
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphPath, pluginName, _config);

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inferInputBlob = inferRequest.GetBlob(inputBlobName);
    auto inputDesc = inferInputBlob->getTensorDesc();
    IE::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inputDesc));
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Get performance
    ASSERT_NO_THROW(dumpPerformance(inferRequest.GetPerformanceCounts()));
}

//------------------------------------------------------------------------------
class InferenceWithCheckLayout : public Inference_onSpecificDevice {
protected:
    void SetUp() override;
};

void InferenceWithCheckLayout::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50.inputPath;
    refOutputPath = PrecompiledResNet_Helper::resnet50.outputPath;

    executableNetwork = ie.ImportNetwork(graphPath, pluginName);
    inferRequest = executableNetwork.CreateInferRequest();
}

TEST_F(InferenceWithCheckLayout, precommit_SyncInferenceAndCheckLayout) {
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inferInputBlob = inferRequest.GetBlob(inputBlobName);
    IE::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inferInputBlob->getTensorDesc()));
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    const auto layoutOrig = inputBlob->getTensorDesc().getLayout();

    ASSERT_NO_THROW(inferRequest.Infer());

    ASSERT_EQ(inferRequest.GetBlob(inputBlobName)->getTensorDesc().getLayout(), layoutOrig);
}

TEST_F(InferenceWithCheckLayout, precommit_CheckInputsLayoutAfterTwoInferences) {
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inferInputBlob = inferRequest.GetBlob(inputBlobName);
    IE::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inferInputBlob->getTensorDesc()));
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    ASSERT_NO_THROW(inferRequest.Infer());

    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto firstOutputBlob = vpu::copyBlob(inferRequest.GetBlob(outputBlobName));

    ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inferInputBlob->getTensorDesc()));
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    ASSERT_NO_THROW(inferRequest.Infer());

    outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto secondOutputBlob = inferRequest.GetBlob(outputBlobName);

    ASSERT_NO_THROW(
        Comparators::compareTopClasses(toFP32(firstOutputBlob), toFP32(secondOutputBlob), numberOfTopClassesToCompare));
}

//------------------------------------------------------------------------------
struct CheckPortsNetworkTestParams final {
    IE::Layout _inputLayout;
    IE::Layout _blobInputLayout;
    bool _importNetwork;

    CheckPortsNetworkTestParams& inputLayout(const IE::Layout& input_layout) {
        this->_inputLayout = input_layout;
        return *this;
    }

    CheckPortsNetworkTestParams& blobInputLayout(const IE::Layout& blob_input_layout) {
        this->_blobInputLayout = blob_input_layout;
        return *this;
    }

    CheckPortsNetworkTestParams& importNetwork(const bool& import_network) {
        this->_importNetwork = import_network;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const CheckPortsNetworkTestParams& p) {
    vpu::formatPrint(os, "[inputLayout:%v, blobInputLayout:%v, importNetwork:%v]", p._inputLayout, p._blobInputLayout,
        p._importNetwork);
    return os;
}

class InferenceCheckPortsNetwork :
    public CoreAPI_Tests,
    public testing::WithParamInterface<CheckPortsNetworkTestParams> {
public:
    std::string modelPath;
    std::string inputPath;
    const size_t inputWidth = 227;
    const size_t inputHeight = 227;
    const size_t numberOfTopClassesToCompare = 3;
    const bool orderedClasses = true;
    const IE::Precision inputPrecision = IE::Precision::U8;

protected:
    void SetUp() override;
};

void InferenceCheckPortsNetwork::SetUp() {
    modelPath =
        ModelsPath() + "/KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10.xml";
    inputPath = "227x227/cat3.bmp";
}

TEST_P(InferenceCheckPortsNetwork, common) {
    const auto& p = GetParam();

    std::cout << "Parameters: " << p << std::endl;

    // --- CNN Network and inputs
    std::cout << "Reading network..." << std::endl;
    ASSERT_NO_THROW(network = ie.ReadNetwork(modelPath));
    IE::InputsDataMap input_info = network.getInputsInfo();
    for (auto& item : input_info) {
        auto input_data = item.second;
        input_data->setLayout(p._inputLayout);
        input_data->setPrecision(inputPrecision);
    }

    if (p._importNetwork) {
        const std::string exportName = "tmpfile";
        IE::ExecutableNetwork exportExecutableNetwork;
        std::cout << "Loading network..." << std::endl;
        ASSERT_NO_THROW(exportExecutableNetwork = ie.LoadNetwork(network, pluginName));
        ASSERT_NO_THROW(exportExecutableNetwork.Export(exportName));
        std::cout << "Importing network..." << std::endl;
        ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(exportName, pluginName));
        ASSERT_TRUE(std::remove(exportName.c_str()) == 0);
    } else {
        std::cout << "Loading network..." << std::endl;
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
    }

    // --- Infer request
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // --- Input Blob
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    IE::Blob::Ptr inputBlob, tmpInputBlob;
    ASSERT_NO_THROW(tmpInputBlob = loadImage(inputPath, inputWidth, inputHeight));
    inputBlob = toLayout(tmpInputBlob, p._blobInputLayout);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // --- Infer
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get result
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    IE::Blob::Ptr outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refInputBlob = toLayout(inputBlob, p._inputLayout);
    IE::Blob::Ptr refOutputBlob;
    ASSERT_NO_THROW(refOutputBlob = ReferenceHelper::CalcCpuReference(network, refInputBlob));
    if (orderedClasses) {
        ASSERT_NO_THROW(
            Comparators::compareTopClasses(toFP32(outputBlob), toFP32(refOutputBlob), numberOfTopClassesToCompare));
    } else {
        ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
            toFP32(outputBlob), toFP32(refOutputBlob), numberOfTopClassesToCompare));
    }
}

const std::vector<CheckPortsNetworkTestParams> checkPortsNetworkParams {
    CheckPortsNetworkTestParams().inputLayout(IE::Layout::NCHW).blobInputLayout(IE::Layout::NCHW).importNetwork(false),

    CheckPortsNetworkTestParams().inputLayout(IE::Layout::NCHW).blobInputLayout(IE::Layout::NHWC).importNetwork(false),

    CheckPortsNetworkTestParams().inputLayout(IE::Layout::NCHW).blobInputLayout(IE::Layout::NCHW).importNetwork(true),

    CheckPortsNetworkTestParams().inputLayout(IE::Layout::NCHW).blobInputLayout(IE::Layout::NHWC).importNetwork(true)};

INSTANTIATE_TEST_CASE_P(CheckPorts, InferenceCheckPortsNetwork, testing::ValuesIn(checkPortsNetworkParams));
