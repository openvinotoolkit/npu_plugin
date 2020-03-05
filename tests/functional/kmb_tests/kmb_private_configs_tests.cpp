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

#include <file_reader.h>
#include <gtest/gtest.h>

#include <allocators.hpp>
#include <ie_core.hpp>
#include <test_model/kmb_test_utils.hpp>

#include "vpu_layers_tests.hpp"

using namespace InferenceEngine;

TEST(KmbPrivateConfigTests, IE_VPU_KMB_SIPP_OUT_COLOR_FORMAT) {
    std::string USE_SIPP = std::getenv("USE_SIPP") != nullptr ? std::getenv("USE_SIPP") : "";
    bool isSIPPEnabled = USE_SIPP.find("1") != std::string::npos;

    if (!isSIPPEnabled) {
        SKIP() << "The test is intended to be run with enviroment USE_SIPP=1";
    }
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork network;
    network = ie.ImportNetwork(modelFilePath, "KMB", {{"VPU_KMB_SIPP_OUT_COLOR_FORMAT", "RGB"}});

    InferenceEngine::InferRequest request;
    request = network.CreateInferRequest();

    std::string inputPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-bgr-nv12.bin";
    const auto inputDims = network.GetInputsInfo().begin()->second->getTensorDesc().getDims();
    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> allocator =
        std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    Blob::Ptr inputBlob = vpu::KmbPlugin::utils::fromNV12File(inputPath, 228, 228, allocator);

    const auto inputName = network.GetInputsInfo().begin()->second->getInputData()->getName();
    PreProcessInfo preProcInfo;
    preProcInfo.setColorFormat(ColorFormat::NV12);
    preProcInfo.setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    request.SetBlob(inputName, inputBlob, preProcInfo);
    request.Infer();
    const auto outputName = network.GetOutputsInfo().begin()->second->getName();

    std::string referenceFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output-228x228-nv12.bin";
    Blob::Ptr outputBlob;
    outputBlob = request.GetBlob(outputName);

    Blob::Ptr referenceBlob = make_shared_blob<float>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 4;
    ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceBlob, NUMBER_OF_CLASSES));
}

static Blob::Ptr createFakeNHWCBlob(const Blob::Ptr& blob) {
    if (blob->getTensorDesc().getLayout() != Layout::NHWC) {
        THROW_IE_EXCEPTION << "fakeNHWCBlob works only with NHWC format";
    }

    if (blob->getTensorDesc().getDims()[1] != 3) {
        THROW_IE_EXCEPTION << "fakeNHWCBlob works only with channels == 3";
    }

    auto tensorDesc = blob->getTensorDesc();
    tensorDesc.setLayout(Layout::NHWC);
    Blob::Ptr fakeNHWC = make_shared_blob<uint8_t>(tensorDesc);
    fakeNHWC->allocate();

    const auto C = tensorDesc.getDims()[1];
    const auto H = tensorDesc.getDims()[2];
    const auto W = tensorDesc.getDims()[3];
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                static_cast<uint8_t*>(fakeNHWC->buffer())[c * H * W + W * h + w] =
                    static_cast<uint8_t*>(blob->buffer())[h * W * C + w * C + c];
            }
        }
    }

    return fakeNHWC;
}

TEST(KmbPrivateConfigTests, FORCE_NCHW_TO_NHWC) {
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork network;
    network = ie.ImportNetwork(modelFilePath, "KMB", {{"VPU_KMB_FORCE_NCHW_TO_NHWC", CONFIG_VALUE(YES)}});

    InferenceEngine::InferRequest request;
    request = network.CreateInferRequest();

    std::string inputPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
    const auto inputTensorDesc = network.GetInputsInfo().begin()->second->getTensorDesc();
    Blob::Ptr inputBlob = make_shared_blob<uint8_t>(inputTensorDesc);
    inputBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(inputPath, inputBlob);

    auto fakeNHWCInput = createFakeNHWCBlob(inputBlob);

    const auto inputName = network.GetInputsInfo().begin()->second->getInputData()->getName();
    request.SetBlob(inputName, fakeNHWCInput);
    request.Infer();

    const auto outputName = network.GetOutputsInfo().begin()->second->getName();
    std::string referenceFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output.bin";
    Blob::Ptr outputBlob;
    outputBlob = request.GetBlob(outputName);

    Blob::Ptr referenceBlob = make_shared_blob<float>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 5;
    ASSERT_NO_THROW(compareTopClasses(toFP32(outputBlob), toFP32(referenceBlob), NUMBER_OF_CLASSES));
}

TEST(KmbPrivateConfigTests, FORCE_2D_TO_NC) {
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork network;
    network = ie.ImportNetwork(modelFilePath, "KMB", {{"VPU_KMB_FORCE_2D_TO_NC", CONFIG_VALUE(YES)}});

    InferenceEngine::InferRequest request;
    request = network.CreateInferRequest();

    std::string inputPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
    const auto inputTensorDesc = network.GetInputsInfo().begin()->second->getTensorDesc();
    Blob::Ptr inputBlob = make_shared_blob<uint8_t>(inputTensorDesc);
    inputBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(inputPath, inputBlob);

    const auto inputName = network.GetInputsInfo().begin()->second->getInputData()->getName();
    request.SetBlob(inputName, inputBlob);
    request.Infer();

    const auto outputName = network.GetOutputsInfo().begin()->second->getName();
    std::string referenceFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output.bin";
    Blob::Ptr outputBlob;
    outputBlob = request.GetBlob(outputName);
    ASSERT_EQ(outputBlob->getTensorDesc().getLayout(), InferenceEngine::Layout::NC);

    Blob::Ptr referenceBlob = make_shared_blob<float>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 5;
    ASSERT_NO_THROW(compareTopClasses(toFP32(outputBlob), toFP32(referenceBlob), NUMBER_OF_CLASSES));
}

// TODO enable when models with FP16 output become available in ModelsPath
TEST(KmbPrivateConfigTests, FORCE_FP16_TO_FP32) {
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork network;
    network = ie.ImportNetwork(modelFilePath, "KMB", {{"VPU_KMB_FORCE_FP16_TO_FP32", CONFIG_VALUE(YES)}});

    InferenceEngine::InferRequest request;
    request = network.CreateInferRequest();

    std::string inputPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
    const auto inputTensorDesc = network.GetInputsInfo().begin()->second->getTensorDesc();
    Blob::Ptr inputBlob = make_shared_blob<uint8_t>(inputTensorDesc);
    inputBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(inputPath, inputBlob);

    const auto inputName = network.GetInputsInfo().begin()->second->getInputData()->getName();
    request.SetBlob(inputName, inputBlob);
    request.Infer();

    const auto outputName = network.GetOutputsInfo().begin()->second->getName();
    std::string referenceFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output.bin";
    Blob::Ptr outputBlob = request.GetBlob(outputName);
    ASSERT_EQ(outputBlob->getTensorDesc().getPrecision(), InferenceEngine::Precision::FP32);

    Blob::Ptr referenceBlob = make_shared_blob<float>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 1;
    ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceBlob, NUMBER_OF_CLASSES));
}
