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
#include <test_model/kmb_test_utils.hpp>

#include "models/model_pooling.h"
#include "vpu_layers_tests.hpp"

using namespace InferenceEngine;
using namespace vpu;

using KmbPrivateConfigTests = vpuLayersTests;

TEST_F(KmbPrivateConfigTests, IE_VPU_KMB_SIPP_OUT_COLOR_FORMAT) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#endif
    std::string USE_SIPP = std::getenv("USE_SIPP") != nullptr ? std::getenv("USE_SIPP") : "";
    bool isSIPPDisabled = USE_SIPP.find("0") != std::string::npos;

    if (isSIPPDisabled) {
        SKIP() << "The test is intended to be run with SIPP enabled";
    }
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    InferenceEngine::ExecutableNetwork network;
    network = core->ImportNetwork(modelFilePath, deviceName, {{"VPU_KMB_SIPP_OUT_COLOR_FORMAT", "RGB"}});

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
    outputBlob = toFP32(request.GetBlob(outputName));

    Blob::Ptr referenceBlob = make_shared_blob<float>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 4;
    ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceBlob, NUMBER_OF_CLASSES));
}

TEST_F(KmbPrivateConfigTests, USE_SIPP) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#endif
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    InferenceEngine::ExecutableNetwork network;
    network = core->ImportNetwork(modelFilePath, deviceName, {{"VPU_KMB_USE_SIPP", CONFIG_VALUE(YES)}});

    InferenceEngine::InferRequest request;
    request = network.CreateInferRequest();

    std::string inputPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-nv12.bin";
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
    outputBlob = toFP32(request.GetBlob(outputName));

    Blob::Ptr referenceBlob = make_shared_blob<float>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 2;
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

TEST_F(KmbPrivateConfigTests, FORCE_NCHW_TO_NHWC) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#endif
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    InferenceEngine::ExecutableNetwork network;
    network = core->ImportNetwork(modelFilePath, deviceName, {{"VPU_KMB_FORCE_NCHW_TO_NHWC", CONFIG_VALUE(YES)}});

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
    Blob::Ptr outputBlob = toFP32(request.GetBlob(outputName));

    Blob::Ptr referenceBlob = make_shared_blob<float>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 5;
    ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceBlob, NUMBER_OF_CLASSES));
}

TEST_F(KmbPrivateConfigTests, SERIALIZE_CNN_BEFORE_COMPILE_FILE) {
#if defined(__arm__) || defined(__aarch64__)
    SKIP();
#endif

    InferenceEngine::ExecutableNetwork network;
    ModelPooling_Helper modelPoolingHelper;
    const std::string testFileName = "tmp_test.xml";
    std::remove(testFileName.c_str());
    for (auto&& input : modelPoolingHelper.network.getInputsInfo()) {
        input.second->setLayout(InferenceEngine::Layout::NHWC);
        input.second->setPrecision(InferenceEngine::Precision::U8);
    }
    for (auto&& output : modelPoolingHelper.network.getOutputsInfo()) {
        output.second->setLayout(InferenceEngine::Layout::NHWC);
        output.second->setPrecision(InferenceEngine::Precision::FP16);
    }
    network = core->LoadNetwork(modelPoolingHelper.network, deviceName);
    std::ifstream notExist(testFileName);
    ASSERT_FALSE(notExist.good());
    network = core->LoadNetwork(modelPoolingHelper.network, deviceName,
        {{"VPU_COMPILER_SERIALIZE_CNN_BEFORE_COMPILE_FILE", testFileName.c_str()}});
    std::ifstream exists(testFileName);
    ASSERT_TRUE(exists.good());
    std::remove(testFileName.c_str());
}
