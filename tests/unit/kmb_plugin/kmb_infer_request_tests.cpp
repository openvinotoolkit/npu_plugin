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

#include "kmb_allocator.h"
#include "kmb_infer_request.h"

namespace ie = InferenceEngine;

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

class MockExecutor : public KmbExecutor {
public:
    MockExecutor(const KmbConfig& config): KmbExecutor(config) {}

    MOCK_METHOD0(deallocateGraph, void());
    MOCK_METHOD1(allocateGraph, void(const std::vector<char>&));
    MOCK_METHOD2(getResult, void(void*, unsigned int));
    MOCK_METHOD2(queueInference, void(void*, size_t));

    MOCK_CONST_METHOD0(getNetworkInputs, ie::InputsDataMap&());
    MOCK_CONST_METHOD0(getNetworkOutputs, ie::OutputsDataMap&());
};

TEST_F(kmbInferRequestConstructionUnitTests, cannotCreateInferRequestWithEmptyInputAndOutput) {
    KmbConfig config;
    auto executor = std::make_shared<MockExecutor>(config);
    KmbInferRequest::Ptr inferRequest;

    ASSERT_THROW(inferRequest = std::make_shared<KmbInferRequest>(InferenceEngine::InputsDataMap(),
                     InferenceEngine::OutputsDataMap(), std::vector<vpu::StageMetaInfo>(), config, executor),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(kmbInferRequestConstructionUnitTests, canCreateInferRequestWithValidParameters) {
    KmbConfig config;

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
        : KmbInferRequest(networkInputs, networkOutputs, blobMetaData, kmbConfig, executor) {};

public:
    MOCK_METHOD6(execSIPPDataPreprocessing,
        void(InferenceEngine::BlobMap&, std::map<std::string, InferenceEngine::PreProcessDataPtr>&,
            InferenceEngine::InputsDataMap&, InferenceEngine::ColorFormat, unsigned int, unsigned int));
    MOCK_METHOD2(execDataPreprocessing, void(InferenceEngine::BlobMap&, bool));
    MOCK_METHOD1(reallocateBlob, ie::Blob::Ptr(const ie::Blob::Ptr&));
    MOCK_CONST_METHOD2(dumpInputBlobHelper, void(const InferenceEngine::Blob::Ptr&, const std::string&));
};

class kmbInferRequestUseCasesUnitTests : public kmbInferRequestConstructionUnitTests {
protected:
    InferenceEngine::InputsDataMap _inputs;
    InferenceEngine::OutputsDataMap _outputs;
    std::shared_ptr<MockExecutor> _executor;
    std::shared_ptr<TestableKmbInferRequest> _inferRequest;

protected:
    void SetUp() override {
        KmbConfig config;
        _executor = std::make_shared<MockExecutor>(config);

        _inputs = setupInputsWithSingleElement();
        _outputs = setupOutputsWithSingleElement();

        ON_CALL(*_executor, getNetworkInputs()).WillByDefault(ReturnRef(_inputs));

        _inferRequest = std::make_shared<TestableKmbInferRequest>(
            _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor);
    }

    ie::NV12Blob::Ptr createNV12Blob(const std::size_t width, const std::size_t height) {
        ie::TensorDesc uvDesc = {ie::Precision::U8, {1, 2, height / 2, width / 2}, ie::Layout::NHWC};
        ie::TensorDesc yDesc = {ie::Precision::U8, {1, 1, height, width}, ie::Layout::NHWC};

        uint8_t* imageData = new uint8_t[height * width / 2 + height * width];

        auto yPlane = ie::make_shared_blob<uint8_t>(yDesc, imageData);
        auto uvPlane = ie::make_shared_blob<uint8_t>(uvDesc, imageData + height * width);

        return ie::make_shared_blob<ie::NV12Blob>(yPlane, uvPlane);
    }

    ie::Blob::Ptr createBlob(const ie::SizeVector dims) {
        if (dims.size() != 4) {
            THROW_IE_EXCEPTION << "Dims size must be 4 for CreateBlob method";
        }
        ie::TensorDesc desc = {ie::Precision::U8, dims, ie::Layout::NHWC};

        auto blob = ie::make_shared_blob<uint8_t>(desc);
        blob->allocate();

        return blob;
    }

    ie::Blob::Ptr createVPUBlob(const ie::SizeVector dims) {
        if (dims.size() != 4) {
            THROW_IE_EXCEPTION << "Dims size must be 4 for createVPUBlob method";
        }

        ie::TensorDesc desc = {ie::Precision::U8, dims, ie::Layout::NHWC};

        auto blob = ie::make_shared_blob<uint8_t>(desc, getKmbAllocator());
        blob->allocate();

        return blob;
    }
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

TEST_F(kmbInferRequestUseCasesUnitTests, requestCopiesNonShareableNV12InputToPreprocWithSIPP) {
    std::string USE_SIPP = std::getenv("USE_SIPP") != nullptr ? std::getenv("USE_SIPP") : "";
    bool isSIPPEnabled = USE_SIPP.find("1") != std::string::npos;

    if (!isSIPPEnabled) {
        SKIP() << "The test is intended to be run with enviroment USE_SIPP=1";
    }

    auto nv12Input = createNV12Blob(1080, 1080);
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
        *dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), execSIPPDataPreprocessing(_, _, _, _, _, _))
        .Times(1);

    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesNonSIPPPPreprocIfResize) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto largeInput = createBlob({dims[0], dims[1], dims[2] * 2, dims[3] * 2});

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    _inferRequest->SetBlob(inputName, largeInput, preProcInfo);

    // TODO: enable this check after execDataPreprocessing become virtual
    // EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(inferRequest.get()), execDataPreprocessing(_, _));
    EXPECT_CALL(*_executor, queueInference(_, _)).Times(1);
    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, requestDumpsBlobIfCorrespondingEnvSet) {
    setenv("IE_VPU_KMB_DUMP_INPUT_PATH", ".", 1 /*overwrite*/);

    ie::Blob::Ptr input;
    auto inputName = _inputs.begin()->first.c_str();
    _inferRequest->GetBlob(inputName, input);
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), dumpInputBlobHelper(input, _))
        .Times(_inputs.size());

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetNV12Blob) {
    std::string USE_SIPP = std::getenv("USE_SIPP") != nullptr ? std::getenv("USE_SIPP") : "";
    bool isSIPPEnabled = USE_SIPP.find("1") != std::string::npos;
    if (!isSIPPEnabled) {
        SKIP() << "The test is intended to be run with enviroment USE_SIPP=1";
    }

    auto nv12Input = createNV12Blob(1080, 1080);
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
    auto inputToSet = createBlob(dims);

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
    auto inputToSet = createBlob(dims);

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
