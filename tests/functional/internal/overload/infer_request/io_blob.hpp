// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <future>

#include "overload/overload_test_utils_vpux.hpp"
#include "shared_test_classes/subgraph/basic_lstm.hpp"

namespace BehaviorTestsDefinitions {

using InferRequestIOBBlobTestVpux = BehaviorTestsUtils::InferRequestTestsVpux;

TEST_P(InferRequestIOBBlobTestVpux, CanCreateInferRequest) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
}

TEST_P(InferRequestIOBBlobTestVpux, failToSetNullptrForInput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, failToSetNullptrForOutput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr outputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, outputBlob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, failToSetUninitializedInputBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, failToSetUninitializedOutputBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, setNotAllocatedInput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
}

TEST_P(InferRequestIOBBlobTestVpux, setNotAllocatedOutput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
}

TEST_P(InferRequestIOBBlobTestVpux, getAfterSetInputDoNotChangeInput) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob = nullptr;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));

    ASSERT_TRUE(actualBlob);
    ASSERT_FALSE(actualBlob->buffer() == nullptr);
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(cnnNet.getInputsInfo().begin()->second->getTensorDesc() == actualBlob->getTensorDesc());
}

TEST_P(InferRequestIOBBlobTestVpux, getAfterSetInputDoNotChangeOutput) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, inputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(actualBlob);
    ASSERT_FALSE(actualBlob->buffer() == nullptr);
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(cnnNet.getOutputsInfo().begin()->second->getTensorDesc() == actualBlob->getTensorDesc());
}

TEST_P(InferRequestIOBBlobTestVpux, failToSetBlobWithIncorrectName) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    const char incorrect_input_name[] = "incorrect_input_name";
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    ASSERT_THROW(req.SetBlob(incorrect_input_name, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, failToSetInputWithIncorrectSizes) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    auto td = cnnNet.getInputsInfo().begin()->second->getTensorDesc();
    auto dims = td.getDims();
    dims[0] *= 2;
    td.reshape(dims);

    InferenceEngine::Blob::Ptr blob = FuncTestUtils::createAndFillBlob(td);
    blob->allocate();
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, failToSetOutputWithIncorrectSizes) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    auto td = cnnNet.getOutputsInfo().begin()->second->getTensorDesc();
    auto dims = td.getDims();
    dims[0] *= 2;
    td.reshape(dims);

    InferenceEngine::Blob::Ptr blob = FuncTestUtils::createAndFillBlob(td);
    blob->allocate();
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, canInferWithoutSetAndGetInOutSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestIOBBlobTestVpux, canInferWithoutSetAndGetInOutAsync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestIOBBlobTestVpux, canProcessDeallocatedInputBlobAfterGetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestIOBBlobTestVpux, canProcessDeallocatedInputBlobAfterGetAndSetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestIOBBlobTestVpux, canProcessDeallocatedInputBlobAfterSetBlobSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, canProcessDeallocatedInputBlobAfterSetBlobAsync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(
            {
                req.StartAsync();
                req.Wait();
            },
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, canProcessDeallocatedOutputBlobAfterSetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
    ASSERT_THROW(
            {
                req.StartAsync();
                req.Wait();
            },
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, canProcessDeallocatedOutputBlobAfterGetAndSetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
    ASSERT_THROW(
            {
                req.StartAsync();
                req.Wait();
            },
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTestVpux, secondCallGetInputDoNotReAllocateData) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTestVpux, secondCallGetOutputDoNotReAllocateData) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTestVpux, secondCallGetInputAfterInferSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW({
        req.StartAsync();
        req.Wait();
    });
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTestVpux, secondCallGetOutputAfterInferSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW({
        req.StartAsync();
        req.Wait();
    });
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTestVpux, canSetInputBlobForInferRequest) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(inputBlob, actualBlob);
}

TEST_P(InferRequestIOBBlobTestVpux, canSetOutputBlobForInferRequest) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> outputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, outputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(outputBlob.get(), actualBlob.get());
}

TEST_P(InferRequestIOBBlobTestVpux, canInferWithSetInOutBlobs) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    InferenceEngine::Blob::Ptr outputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestIOBBlobTestVpux, canInferWithGetIn) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    ASSERT_NO_THROW(req.Infer());
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW({
        req.StartAsync();
        sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    });
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

TEST_P(InferRequestIOBBlobTestVpux, canInferWithGetOut) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    ASSERT_NO_THROW(req.Infer());
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW({
        req.StartAsync();
        sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    });
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

TEST_P(InferRequestIOBBlobTestVpux, canReallocateExternalBlobViaGet) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 10, 10});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("relu");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        result->set_friendly_name("result");

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto inBlob = req.GetBlob("param");
    auto outBlob = req.GetBlob("relu");
    inBlob->allocate();
    outBlob->allocate();

    ASSERT_NO_THROW(req.Infer());
}

}  // namespace BehaviorTestsDefinitions