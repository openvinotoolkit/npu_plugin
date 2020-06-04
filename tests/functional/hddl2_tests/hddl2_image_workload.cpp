//
// Copyright 2019 Intel Corporation.
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

#include <helper_ie_core.h>

#include <blob_factory.hpp>
#include <ie_core.hpp>

#include "comparators.h"
#include "creators/creator_blob_nv12.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "ie_blob.h"
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

class ImageWorkload_Tests : public IE_Core_Helper, public ::testing::Test {
public:
    std::string graphPath;
    std::string refInputPath;
    std::string refOutputPath;

    const size_t numberOfTopClassesToCompare = 5;

protected:
    void SetUp() override;
};

void ImageWorkload_Tests::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50.inputPath;
    refOutputPath = PrecompiledResNet_Helper::resnet50.outputPath;
}

//------------------------------------------------------------------------------
using ImageWorkload_WithoutPreprocessing = ImageWorkload_Tests;
TEST_F(ImageWorkload_WithoutPreprocessing, SyncInference) {
    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Import or load network
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphPath, "HDDL2");

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = inferRequest.GetBlob(inputBlobName);
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, inputBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto refBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    refBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, refBlob));

    // --- Look at raw input/output/reference
    const size_t byteToPrint = 8;
    EXPECT_NO_THROW(printRawBlob(inputBlob, byteToPrint, "inputBlob"));
    EXPECT_NO_THROW(printRawBlob(outputBlob, byteToPrint, "outputBlob"));
    EXPECT_NO_THROW(printRawBlob(refBlob, byteToPrint, "refBlob"));

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == IE::Precision::U8);
    ASSERT_NO_THROW(Comparators::compareTopClasses(outputBlob, refBlob, numberOfTopClassesToCompare));
}

//------------------------------------------------------------------------------
class ImageWorkload_WithPreprocessing : public ImageWorkload_Tests {
protected:
    void SetUp() override;
};

void ImageWorkload_WithPreprocessing::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50_dpu.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50_dpu.nv12Input;
    refOutputPath = PrecompiledResNet_Helper::resnet50_dpu.nv12Output;
}

TEST_F(ImageWorkload_WithPreprocessing, SyncInference) {
    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Import or load network
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphPath, "HDDL2");

    // ---- Create infer request
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Load NV12 Image and create blob from it
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;

    // TODO Fix to follow same approach as hello nv12 classification sample
    const size_t image_width = 228;
    const size_t image_height = 228;
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(refInputPath, image_width, image_height);

    // Since it 228x228 image on 224x224 network, resize preprocessing also required
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(IE::RESIZE_BILINEAR);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, nv12InputBlob, preprocInfo);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto refBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    refBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, refBlob));

    // --- Look at raw output/reference
    const size_t byteToPrint = 8;
    EXPECT_NO_THROW(printRawBlob(outputBlob, byteToPrint, "outputBlob"));
    EXPECT_NO_THROW(printRawBlob(refBlob, byteToPrint, "refBlob"));

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == IE::Precision::U8);
    ASSERT_NO_THROW(Comparators::compareTopClasses(outputBlob, refBlob, numberOfTopClassesToCompare));
}
