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

#include <format_reader_ptr.h>
#include <helper_ie_core.h>

#include <blob_factory.hpp>
#include <vpux_private_config.hpp>

#include "comparators.h"
#include "creators/creator_blob_nv12.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "ie_blob.h"
#include "ie_utils.hpp"
#include "models/precompiled_resnet.h"
#include <helper_calc_cpu_ref.h>
#include "tests_common.hpp"

namespace IE = InferenceEngine;

class ImageWorkload_Tests : public ::testing::Test {
public:
    std::string graphPath;
    std::string modelPath;

    const size_t inputWidth = 224;
    const size_t inputHeight = 224;
    const size_t numberOfTopClassesToCompare = 3;

protected:
    void SetUp() override;
};

void ImageWorkload_Tests::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
    modelPath = PrecompiledResNet_Helper::resnet50.modelPath;
}

//------------------------------------------------------------------------------
using ImageWorkload_WithoutPreprocessing = ImageWorkload_Tests;
// Test fails with error:
// C++ exception with description "[PARAMETER_MISMATCH] Failed to set input blob.
// Blocking descriptor mismatch." thrown in the test body.
// [Track number: S#42277]
TEST_F(ImageWorkload_WithoutPreprocessing, DISABLED_precommit_SyncInference) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphPath, "VPUX");

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = IE_Core_Helper::loadCatImage();
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}

// Test fails with error:
// kmb-plugin/tests_deprecated/functional/hddl2_tests/hddl2_image_workload.cpp:95: Failure
// Expected: inferRequest = executableNetwork.CreateInferRequest() doesn't throw an exception.
// Actual: it throws:Can't create infer request!
// Please make sure that the device is available. Only exports can be made.
// [Track number: S#42484]
TEST_F(ImageWorkload_WithoutPreprocessing, DISABLED_precommit_SyncInferenceNCHWInput) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphPath, "VPUX");

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;

    // Load image in different layout to validate repacking
    auto inputBlob = IE_Core_Helper::loadCatImage(IE::Layout::NCHW);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}

//------------------------------------------------------------------------------
class ImageWorkload_WithPreprocessing : public ImageWorkload_Tests {
public:
    const size_t numberOfTopClassesToCompare = 3;

    std::string inputNV12Path;

protected:
    void SetUp() override;
};

void ImageWorkload_WithPreprocessing::SetUp() {
    ImageWorkload_Tests::SetUp();
    inputNV12Path = TestDataHelpers::get_data_path() + "/" + std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "/cat3.yuv";
}

// Test fails with error:
// [ RUN      ] ImageWorkload_WithPreprocessing.precommit_SyncInference
// kmb-plugin/tests_deprecated/functional/hddl2_tests/hddl2_image_workload.cpp:144: Failure
// Expected: inferRequest = executableNetwork.CreateInferRequest() doesn't throw an exception.
// Actual: it throws:Can't create infer request!
// Please make sure that the device is available. Only exports can be made.
// [  FAILED  ] ImageWorkload_WithPreprocessing.precommit_SyncInference (260 ms)
// [Track number: S#42486]
TEST_F(ImageWorkload_WithPreprocessing, DISABLED_precommit_SyncInference) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphPath, "VPUX");

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Load NV12 Image and create blob from it
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;

    // TODO Fix to follow same approach as hello nv12 classification sample
    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load NV12 input
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(
        inputNV12Path, inputWidth, inputHeight);

    // Preprocessing
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, nv12InputBlob, preprocInfo);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, nv12InputBlob, &preprocInfo);

    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}

TEST_F(ImageWorkload_WithPreprocessing, precommit_SyncInference_RGBToBGR) {
    // ---- Load inference engine instance
    IE::Core ie;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphPath, "VPUX");

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set RGB input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    const auto isBGR = false;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", 224, 224, IE::NCHW, isBGR);

    // ---- Preprocessing
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::PreProcessInfo preProcInfo = inferRequest.GetPreProcess(inputName);
    preProcInfo.setColorFormat(IE::ColorFormat::RGB);

    // ---- Set RGB blob with preprocessing information
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob, preProcInfo));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}
