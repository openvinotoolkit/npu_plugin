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

#include "vpu_layers_tests.hpp"

using namespace InferenceEngine;

TEST(KmbPrivateConfigTests, IE_VPU_KMB_SIPP_OUT_COLOR_FORMAT) {
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2-dpu/mobilenet-v2-dpu.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork network;
    network = ie.ImportNetwork(modelFilePath, "KMB", {{"VPU_KMB_SIPP_OUT_COLOR_FORMAT", "RGB"}});

    InferenceEngine::InferRequest request;
    request = network.CreateInferRequest();

    std::string inputPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2-dpu/input-228x228-bgr-nv12.bin";
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

    std::string referenceFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2-dpu/output-228x228-nv12.bin";
    Blob::Ptr outputBlob;
    outputBlob = request.GetBlob(outputName);

    Blob::Ptr referenceBlob = make_shared_blob<uint8_t>(outputBlob->getTensorDesc());
    referenceBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(referenceFilePath, referenceBlob);

    const size_t NUMBER_OF_CLASSES = 4;
    ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceBlob, NUMBER_OF_CLASSES));
}