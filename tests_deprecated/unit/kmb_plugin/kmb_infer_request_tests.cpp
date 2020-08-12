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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vpux.hpp>

#include "creators/creator_blob.h"
#include "creators/creator_blob_nv12.h"
#include "kmb_infer_request.h"
#include "kmb_private_config.hpp"

namespace ie = InferenceEngine;

using namespace ::testing;
using namespace vpu::KmbPlugin;

class kmbInferRequestConstructionUnitTests : public ::testing::Test {
protected:
    ie::InputsDataMap setupInputsWithSingleElement() {
        std::string inputName = "input";
        ie::TensorDesc inputDescription = ie::TensorDesc(ie::Precision::U8, {1, 3, 224, 224}, ie::Layout::NHWC);
        ie::DataPtr inputData = std::make_shared<ie::Data>(inputName, inputDescription);
        ie::InputInfo::Ptr inputInfo = std::make_shared<ie::InputInfo>();
        inputInfo->setInputData(inputData);
        ie::InputsDataMap inputs = {{inputName, inputInfo}};

        return inputs;
    }

    ie::OutputsDataMap setupOutputsWithSingleElement() {
        std::string outputName = "output";
        ie::TensorDesc outputDescription = ie::TensorDesc(ie::Precision::U8, {1000}, ie::Layout::C);
        ie::DataPtr outputData = std::make_shared<ie::Data>(outputName, outputDescription);
        ie::OutputsDataMap outputs = {{outputName, outputData}};

        return outputs;
    }
};

constexpr int defaultDeviceId = 0;
class MockNetworkDescription : public vpux::NetworkDescription {
    const vpux::DataMap& getInputsInfo() const override { return inputs; }

    const vpux::DataMap& getOutputsInfo() const override { return outputs; }

    const vpux::DataMap& getDeviceInputsInfo() const override { return inputs; }

    const vpux::DataMap& getDeviceOutputsInfo() const override { return outputs; }

    const std::vector<char>& getCompiledNetwork() const override { return network; }

    const std::string& getName() const override { return name; }

private:
    std::string name;
    vpux::DataMap inputs;
    vpux::DataMap outputs;
    std::vector<char> network;
};

class MockExecutor : public vpux::Executor {
public:
    MOCK_METHOD1(push, void(const ie::BlobMap&));
    MOCK_METHOD1(pull, void(ie::BlobMap&));

    void setup(const InferenceEngine::ParamMap&) {}

    bool isPreProcessingSupported(const InferenceEngine::PreProcessInfo&) { return false; }
    std::map<std::string, ie::InferenceEngineProfileInfo> getLayerStatistics() {
        return std::map<std::string, ie::InferenceEngineProfileInfo>();
    }

    ie::Parameter getParameter(const std::string&) { return ie::Parameter(); }
};

TEST_F(kmbInferRequestConstructionUnitTests, cannotCreateInferRequestWithEmptyInputAndOutput) {
    KmbConfig config;

    auto executor = std::make_shared<MockExecutor>();
    KmbInferRequest::Ptr inferRequest;

    auto allocator = std::make_shared<KmbAllocator>(defaultDeviceId);
    ASSERT_THROW(inferRequest = std::make_shared<KmbInferRequest>(ie::InputsDataMap(), ie::OutputsDataMap(),
                     std::vector<vpu::StageMetaInfo>(), config, executor, allocator),
        ie::details::InferenceEngineException);
}

TEST_F(kmbInferRequestConstructionUnitTests, canCreateInferRequestWithValidParameters) {
    KmbConfig config;
    auto executor = std::make_shared<MockExecutor>();
    auto inputs = setupInputsWithSingleElement();
    auto outputs = setupOutputsWithSingleElement();

    auto allocator = std::make_shared<KmbAllocator>(defaultDeviceId);
    KmbInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(inferRequest = std::make_shared<KmbInferRequest>(
                        inputs, outputs, std::vector<vpu::StageMetaInfo>(), config, executor, allocator));
}
class TestableKmbInferRequest : public KmbInferRequest {
public:
    TestableKmbInferRequest(const ie::InputsDataMap& networkInputs, const ie::OutputsDataMap& networkOutputs,
        const std::vector<vpu::StageMetaInfo>& blobMetaData, const KmbConfig& kmbConfig,
        const std::shared_ptr<vpux::Executor>& executor, const std::shared_ptr<ie::IAllocator>& allocator)
        : KmbInferRequest(networkInputs, networkOutputs, blobMetaData, kmbConfig, executor, allocator){};

public:
    MOCK_METHOD6(execKmbDataPreprocessing, void(ie::BlobMap&, std::map<std::string, ie::PreProcessDataPtr>&,
                                               ie::InputsDataMap&, ie::ColorFormat, unsigned int, unsigned int));
    MOCK_METHOD2(execDataPreprocessing, void(ie::BlobMap&, bool));
};

// FIXME: cannot be run on x86 the tests below use vpusmm allocator and requires vpusmm driver instaled
// can be enabled with other allocator
// [Track number: S#28136]
#if defined(__arm__) || defined(__aarch64__)

class kmbInferRequestUseCasesUnitTests : public kmbInferRequestConstructionUnitTests {
protected:
    ie::InputsDataMap _inputs;
    ie::OutputsDataMap _outputs;
    std::shared_ptr<MockExecutor> _executor;
    std::shared_ptr<TestableKmbInferRequest> _inferRequest;

protected:
    void SetUp() override {
        KmbConfig config;

        _executor = std::make_shared<MockExecutor>();

        _inputs = setupInputsWithSingleElement();
        _outputs = setupOutputsWithSingleElement();

        _allocator = std::make_shared<KmbAllocator>(defaultDeviceId);
        _inferRequest = std::make_shared<TestableKmbInferRequest>(
            _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor, _allocator);
    }

    ie::Blob::Ptr createVPUBlob(const ie::SizeVector dims, const ie::Layout layout = ie::Layout::NHWC) {
        if (dims.size() != 4) {
            THROW_IE_EXCEPTION << "Dims size must be 4 for createVPUBlob method";
        }

        ie::TensorDesc desc = {ie::Precision::U8, dims, layout};

        auto blob = ie::make_shared_blob<uint8_t>(desc, _allocator);
        blob->allocate();

        return blob;
    }

    ie::NV12Blob::Ptr createNV12VPUBlob(const std::size_t width, const std::size_t height) {
        nv12Data = reinterpret_cast<uint8_t*>(_allocator->alloc(height * width * 3 / 2));
        return NV12Blob_Creator::createFromMemory(width, height, nv12Data);
    }

    void TearDown() override {
        // nv12Data can be allocated in two different ways in the tests below
        // that why we need to branches to handle removing of memory
        if (nv12Data != nullptr) {
            if (_allocator->isValidPtr(nv12Data)) {
                _allocator->free(nv12Data);
            }
        }
    }

private:
    uint8_t* nv12Data = nullptr;
    std::shared_ptr<KmbAllocator> _allocator;
};

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesTheSameInputForInferenceAsGetBlobReturns) {
    auto inputName = _inputs.begin()->first.c_str();

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);
    ie::BlobMap inputs = {{inputName, input}};
    EXPECT_CALL(*_executor, push(inputs)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesExternalShareableBlobForInference) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto vpuBlob = createVPUBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    ie::BlobMap inputs = {{inputName, vpuBlob}};
    EXPECT_CALL(*_executor, push(inputs)).Times(1);

    _inferRequest->SetBlob(inputName, vpuBlob);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesNonSIPPPPreprocIfResize) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto largeInput = Blob_Creator::createBlob({dims[0], dims[1], dims[2] * 2, dims[3] * 2});

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    _inferRequest->SetBlob(inputName, largeInput, preProcInfo);

    // TODO: enable this check after execDataPreprocessing become virtual
    // EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(inferRequest.get()), execDataPreprocessing(_, _));
    EXPECT_CALL(*_executor, push(_)).Times(1);
    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetNV12Blob) {
    auto nv12Input = NV12Blob_Creator::createBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);
    ASSERT_EQ(nv12Input->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetVPUBlob) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto vpuInput = createVPUBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    _inferRequest->SetBlob(inputName, vpuInput);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);

    ASSERT_EQ(vpuInput->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetLargeVPUBlob) {
    auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    dims[2] *= 2;
    dims[3] *= 2;
    auto vpuInput = createVPUBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    _inferRequest->SetBlob(inputName, vpuInput, preProcInfo);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);

    ASSERT_EQ(vpuInput->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetOrdinaryBlobMatchedNetworkInput) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto inputToSet = Blob_Creator::createBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    _inferRequest->SetBlob(inputName, inputToSet);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);

    ASSERT_EQ(inputToSet->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetOrdinaryBlobNotMatchedNetworkInput) {
    auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    dims[2] *= 2;
    dims[3] *= 2;
    auto inputToSet = Blob_Creator::createBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    _inferRequest->SetBlob(inputName, inputToSet, preProcInfo);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);

    ASSERT_EQ(inputToSet->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, BGRIsDefaultColorFormatForSIPPPreproc) {
    auto nv12Input = createNV12VPUBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()),
        execKmbDataPreprocessing(_, _, _, ie::ColorFormat::BGR, _, _));

    _inferRequest->InferAsync();
}

class kmbInferRequestOutColorFormatSIPPUnitTests :
    public kmbInferRequestUseCasesUnitTests,
    public testing::WithParamInterface<const char*> {};

TEST_P(kmbInferRequestOutColorFormatSIPPUnitTests, preprocessingUseRGBIfConfigIsSet) {
    KmbConfig config;
    const auto configValue = GetParam();
    config.update({{VPU_KMB_CONFIG_KEY(SIPP_OUT_COLOR_FORMAT), configValue}});

    auto allocator = std::make_shared<KmbAllocator>(defaultDeviceId);
    _inferRequest = std::make_shared<TestableKmbInferRequest>(
        _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor, allocator);

    auto nv12Input = createNV12VPUBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    auto expectedColorFmt = [](const std::string colorFmt) {
        if (colorFmt == "RGB") {
            return ie::ColorFormat::RGB;
        } else if (colorFmt == "BGR") {
            return ie::ColorFormat::BGR;
        }

        return ie::ColorFormat::RAW;
    };
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()),
        execKmbDataPreprocessing(_, _, _, expectedColorFmt(configValue), _, _));

    _inferRequest->InferAsync();
}

INSTANTIATE_TEST_CASE_P(
    SupportedColorFormats, kmbInferRequestOutColorFormatSIPPUnitTests, testing::Values("RGB", "BGR"));

class kmbInferRequestSIPPPreprocessing :
    public kmbInferRequestUseCasesUnitTests,
    public testing::WithParamInterface<std::string> {};

TEST_P(kmbInferRequestSIPPPreprocessing, canDisableSIPP) {
    KmbConfig config;
    const auto param = GetParam();
    if (param == "config_option") {
        config.update({{"VPU_KMB_USE_SIPP", CONFIG_VALUE(NO)}});
    } else if (param == "environment_variable") {
        setenv("USE_SIPP", "0", 1);
    }

    auto allocator = std::make_shared<KmbAllocator>(defaultDeviceId);
    _inferRequest = std::make_shared<TestableKmbInferRequest>(
        _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor, allocator);

    auto nv12Input = createNV12VPUBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(
        *dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), execKmbDataPreprocessing(_, _, _, _, _, _))
        .Times(0);

    _inferRequest->InferAsync();

    if (param == "environment_variable") {
        unsetenv("USE_SIPP");
    }
}

INSTANTIATE_TEST_CASE_P(
    WaysToDisableSIPP, kmbInferRequestSIPPPreprocessing, testing::Values("environment_variable", "config_option"));

#endif  //  __arm__
