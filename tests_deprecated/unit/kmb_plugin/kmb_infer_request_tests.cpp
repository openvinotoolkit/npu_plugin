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

#include "creators/creator_blob.h"
#include "creators/creator_blob_nv12.h"
#include "kmb_allocator.h"
#include "kmb_infer_request.h"
#include "kmb_private_config.hpp"

using namespace ::testing;
using namespace vpu::KmbPlugin;

class kmbInferRequestConstructionUnitTests : public ::testing::Test {
protected:
    InferenceEngine::InputsDataMap setupInputsWithSingleElement() {
        std::string inputName = "input";
        ie::TensorDesc inputDescription = ie::TensorDesc(ie::Precision::U8, {1, 3, 224, 224}, ie::Layout::NHWC);
        ie::DataPtr inputData = std::make_shared<ie::Data>(inputName, inputDescription);
        ie::InputInfo::Ptr inputInfo = std::make_shared<ie::InputInfo>();
        inputInfo->setInputData(inputData);
        InferenceEngine::InputsDataMap inputs = {{inputName, inputInfo}};

        return inputs;
    }

    InferenceEngine::OutputsDataMap setupOutputsWithSingleElement() {
        std::string outputName = "output";
        ie::TensorDesc outputDescription = ie::TensorDesc(ie::Precision::U8, {1000}, ie::Layout::C);
        ie::DataPtr outputData = std::make_shared<ie::Data>(outputName, outputDescription);
        InferenceEngine::OutputsDataMap outputs = {{outputName, outputData}};

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

class MockExecutor : public KmbExecutor {
public:
    MockExecutor(const KmbConfig& config)
        : KmbExecutor(
              std::make_shared<MockNetworkDescription>(), std::make_shared<KmbAllocator>(defaultDeviceId), config) {}

    MOCK_METHOD1(allocateGraph, void(const std::string&));
    MOCK_METHOD0(deallocateGraph, void());
    MOCK_METHOD4(
        allocateGraph, void(const std::vector<char>&, const ie::InputsDataMap&, const ie::OutputsDataMap&, bool));
    MOCK_METHOD2(getResult, void(void*, unsigned int));
    MOCK_METHOD2(queueInference, void(void*, size_t));

    MOCK_CONST_METHOD0(getDeviceInputs, vpux::DataMap&());
    MOCK_CONST_METHOD0(getDeviceOutputs, vpux::DataMap&());
};

TEST_F(kmbInferRequestConstructionUnitTests, cannotCreateInferRequestWithEmptyInputAndOutput) {
    KmbConfig config;
    config.update({{"VPU_KMB_KMB_EXECUTOR", "NO"}});

    auto executor = std::make_shared<MockExecutor>(config);
    KmbInferRequest::Ptr inferRequest;

    ASSERT_THROW(inferRequest = std::make_shared<KmbInferRequest>(InferenceEngine::InputsDataMap(),
                     InferenceEngine::OutputsDataMap(), std::vector<vpu::StageMetaInfo>(), config, executor),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(kmbInferRequestConstructionUnitTests, canCreateInferRequestWithValidParameters) {
    KmbConfig config;
    config.update({{"VPU_KMB_KMB_EXECUTOR", "NO"}});

    auto executor = std::make_shared<MockExecutor>(config);
    auto inputs = setupInputsWithSingleElement();
    auto outputs = setupOutputsWithSingleElement();

    KmbInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(inferRequest = std::make_shared<KmbInferRequest>(
                        inputs, outputs, std::vector<vpu::StageMetaInfo>(), config, executor));
}

class TestableKmbInferRequest : public KmbInferRequest {
public:
    TestableKmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
        const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<vpu::StageMetaInfo>& blobMetaData,
        const KmbConfig& kmbConfig, const KmbExecutor::Ptr& executor)
        : KmbInferRequest(networkInputs, networkOutputs, blobMetaData, kmbConfig, executor){};

public:
    MOCK_METHOD6(execKmbDataPreprocessing,
        void(InferenceEngine::BlobMap&, std::map<std::string, InferenceEngine::PreProcessDataPtr>&,
            InferenceEngine::InputsDataMap&, InferenceEngine::ColorFormat, unsigned int, unsigned int));
    MOCK_METHOD2(execDataPreprocessing, void(InferenceEngine::BlobMap&, bool));
    MOCK_METHOD1(reallocateBlob, ie::Blob::Ptr(const ie::Blob::Ptr&));
};

// FIXME: cannot be run on x86 the tests below use vpusmm allocator and requires vpusmm driver instaled
// can be enabled with other allocator
// [Track number: S#28136]
#if defined(__arm__) || defined(__aarch64__)

class kmbInferRequestUseCasesUnitTests : public kmbInferRequestConstructionUnitTests {
protected:
    InferenceEngine::InputsDataMap _inputs;
    vpux::DataMap _deviceInputs;
    InferenceEngine::OutputsDataMap _outputs;
    std::shared_ptr<MockExecutor> _executor;
    std::shared_ptr<TestableKmbInferRequest> _inferRequest;

protected:
    void SetUp() override {
        KmbConfig config;
        config.update({{VPU_KMB_CONFIG_KEY(KMB_EXECUTOR), "NO"}});

        _executor = std::make_shared<MockExecutor>(config);

        _inputs = setupInputsWithSingleElement();
        _outputs = setupOutputsWithSingleElement();

        _deviceInputs.insert({{_inputs.begin()->first, _inputs.begin()->second->getInputData()}});
        ON_CALL(*_executor, getDeviceInputs()).WillByDefault(ReturnRef(_deviceInputs));

        _inferRequest = std::make_shared<TestableKmbInferRequest>(
            _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor);

        _allocatorPtr = std::make_shared<KmbAllocator>(defaultDeviceId);
    }

    ie::Blob::Ptr createVPUBlob(const ie::SizeVector dims, const ie::Layout layout = ie::Layout::NHWC) {
        if (dims.size() != 4) {
            THROW_IE_EXCEPTION << "Dims size must be 4 for createVPUBlob method";
        }

        ie::TensorDesc desc = {ie::Precision::U8, dims, layout};

        auto blob = ie::make_shared_blob<uint8_t>(desc, _allocatorPtr);
        blob->allocate();

        return blob;
    }

    ie::NV12Blob::Ptr createNV12VPUBlob(const std::size_t width, const std::size_t height) {
        nv12Data = reinterpret_cast<uint8_t*>(_allocatorPtr->alloc(height * width * 3 / 2));
        return NV12Blob_Creator::createFromMemory(width, height, nv12Data);
    }

    void TearDown() override {
        // nv12Data can be allocated in two different ways in the tests below
        // that why we need to branches to handle removing of memory
        if (nv12Data != nullptr) {
            if (_allocatorPtr->isValidPtr(nv12Data)) {
                _allocatorPtr->free(nv12Data);
            }
        }
    }

private:
    uint8_t* nv12Data = nullptr;
    std::shared_ptr<KmbAllocator> _allocatorPtr;
};

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesTheSameInputForInferenceAsGetBlobReturns) {
    auto inputName = _inputs.begin()->first.c_str();

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);
    auto buffer = input->buffer().as<void*>();
    EXPECT_CALL(*_executor, queueInference(buffer, input->byteSize())).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, requestCopiesNonShareableInputToInfer) {
    const auto inputDesc = _inputs.begin()->second->getTensorDesc();
    ie::Blob::Ptr input = ie::make_shared_blob<uint8_t>(inputDesc);
    input->allocate();

    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto reallocatedInput = createVPUBlob(dims);
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), reallocateBlob(input))
        .Times(1)
        .WillOnce(Return(reallocatedInput));

    auto inputName = _inputs.begin()->first.c_str();
    auto buffer = reallocatedInput->buffer().as<void*>();
    EXPECT_CALL(*_executor, queueInference(buffer, reallocatedInput->byteSize())).Times(1);

    _inferRequest->SetBlob(inputName, input);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesExternalShareableBlobForInference) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto vpuBlob = createVPUBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    auto buffer = vpuBlob->buffer().as<void*>();
    EXPECT_CALL(*_executor, queueInference(buffer, vpuBlob->byteSize())).Times(1);

    _inferRequest->SetBlob(inputName, vpuBlob);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

// tracking number: S#32515
TEST_F(kmbInferRequestUseCasesUnitTests, requestCopiesNonShareableNV12InputToPreprocWithSIPP) {
    auto nv12Input = NV12Blob_Creator::createBlob(1080, 1080);
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), reallocateBlob(nv12Input->uv()))
        .Times(1)
        .WillOnce(Return(nv12Input->uv()));
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), reallocateBlob(nv12Input->y()))
        .Times(1)
        .WillOnce(Return(nv12Input->y()));

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(
        *dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), execKmbDataPreprocessing(_, _, _, _, _, _))
        .Times(1);

    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);

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
    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);
    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

// tracking number: S#32515
TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetNV12Blob) {
    auto nv12Input = NV12Blob_Creator::createBlob(1080, 1080);
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), reallocateBlob(nv12Input->uv()))
        .Times(1)
        .WillOnce(Return(nv12Input->uv()));
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), reallocateBlob(nv12Input->y()))
        .Times(1)
        .WillOnce(Return(nv12Input->y()));

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);

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

    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);

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

    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);

    ASSERT_EQ(vpuInput->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetOrdinaryBlobMatchedNetworkInput) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto inputToSet = Blob_Creator::createBlob(dims);

    auto reallocatedInput = createVPUBlob(dims);
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), reallocateBlob(inputToSet))
        .Times(1)
        .WillOnce(Return(reallocatedInput));

    auto inputName = _inputs.begin()->first.c_str();
    _inferRequest->SetBlob(inputName, inputToSet);

    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);

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

    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input;
    _inferRequest->GetBlob(inputName, input);

    ASSERT_EQ(inputToSet->buffer().as<void*>(), input->buffer().as<void*>());
}
// tracking number: S#32515
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
// tracking number: S#32515
TEST_P(kmbInferRequestOutColorFormatSIPPUnitTests, preprocessingUseRGBIfConfigIsSet) {
    KmbConfig config;
    const auto configValue = GetParam();
    config.update({{VPU_KMB_CONFIG_KEY(SIPP_OUT_COLOR_FORMAT), configValue}});

    _inferRequest = std::make_shared<TestableKmbInferRequest>(
        _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor);

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

    _inferRequest = std::make_shared<TestableKmbInferRequest>(
        _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor);

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
