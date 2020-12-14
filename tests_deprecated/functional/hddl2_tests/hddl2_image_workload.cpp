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

#include <helper_ie_core.h>

#include <blob_factory.hpp>
#include <vpux_private_config.hpp>
#include "executable_network_factory.h"

#include "comparators.h"
#include "creators/creator_blob_nv12.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "ie_blob.h"
#include "ie_utils.hpp"
#include <helper_calc_cpu_ref.h>
#include "tests_common.hpp"
#include "models/models_constant.h"

namespace IE = InferenceEngine;

class ImageWorkload_Tests : public ::testing::Test {
public:
    Models::ModelDesc modelForNoPreprocess = Models::squeezenet1_1;
    // TODO For preprocessing 224x224 model required (or width and height % 2 == 0)
    Models::ModelDesc modelForPreprocessing = Models::googlenet_v1;

    const size_t numberOfTopClassesToCompare = 3;
};

//------------------------------------------------------------------------------
using ImageWorkload_WithoutPreprocessing = ImageWorkload_Tests;
TEST_F(ImageWorkload_WithoutPreprocessing, precommit_SyncInference) {
    const Models::ModelDesc modelToUse = modelForNoPreprocess;

    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", modelToUse.width, modelToUse.height, IE::NHWC, true);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}

/** @brief Validate repacking from NCHW to NHWC */
TEST_F(ImageWorkload_WithoutPreprocessing, precommit_SyncInferenceNCHWInput) {
    const Models::ModelDesc modelToUse = modelForNoPreprocess;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;

    // Load image in different layout to validate repacking
    auto inputNCHWBlob = IE_Core_Helper::loadImage("cat3.bmp", modelToUse.width, modelToUse.height, IE::NCHW, true);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputNCHWBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto inputNHWCBlob = IE_Core_Helper::loadImage("cat3.bmp", modelToUse.width, modelToUse.height, IE::NHWC, true);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputNHWCBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}

//------------------------------------------------------------------------------
class ImageWorkload_WithPreprocessing : public ImageWorkload_Tests {
public:
    const std::string inputNV12Path = TestDataHelpers::get_data_path() + "/" +
                                      std::to_string(modelForPreprocessing.width) + "x" +
                                      std::to_string(modelForPreprocessing.height) + "/cat3.yuv";
};

TEST_F(ImageWorkload_WithPreprocessing, precommit_SyncInference) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Load NV12 Image and create blob from it
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;

    // TODO Fix to follow same approach as hello nv12 classification sample
    // ---- Load frame to remote memory (emulate VAAPI result)
    // ----- Load NV12 input
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(
                                        inputNV12Path, modelToUse.width, modelToUse.height);

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
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, nv12InputBlob, &preprocInfo);

    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}

TEST_F(ImageWorkload_WithPreprocessing, precommit_SyncInference_RGBToBGR) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set RGB input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    const auto isBGR = false;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", modelToUse.width, modelToUse.height, IE::NHWC, isBGR);

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

    // --- Reference Blob - Same image, BGR instead of RGB
    const bool isBGRforCPU = true;
    auto inputBlobCPU = IE_Core_Helper::loadImage("cat3.bmp", modelToUse.width, modelToUse.height, IE::NHWC, isBGRforCPU);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlobCPU);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}

//------------------------------------------------------------------------------
class ImageWorkload_SpecificCases : public ImageWorkload_Tests {
public:
    const std::string inputNV12Path = TestDataHelpers::get_data_path() + "/" +
                                      std::to_string(modelForPreprocessing.width) + "x" +
                                      std::to_string(modelForPreprocessing.height) + "/cat3.yuv";
};

/** @brief Execute inference with preprocessing and after that without preprocessing */
TEST_F(ImageWorkload_SpecificCases, precommit_WithoutPreprocessingAndPreprocessing) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;

    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // ---- Without preprocessing - set blob
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", modelToUse.width, modelToUse.height, IE::NHWC, true);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Without preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);
    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));


    // ---- With preprocessing - set blob
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(inputNV12Path, modelToUse.width, modelToUse.height);
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);
    inferRequest.SetBlob(inputName, nv12InputBlob, preprocInfo);

    // ---- With preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    outputBlob = inferRequest.GetBlob(outputBlobName);
    refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, nv12InputBlob, &preprocInfo);
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));;
}

/** @brief Execute inference without preprocessing and after that with preprocessing  */
TEST_F(ImageWorkload_SpecificCases, precommit_PreprocessingAndWithoutPreprocessing) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;

    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // ---- With preprocessing - set blob
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(inputNV12Path, modelToUse.width, modelToUse.height);
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);
    inferRequest.SetBlob(inputName, nv12InputBlob, preprocInfo);

    // ---- With preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);
    auto refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, nv12InputBlob, &preprocInfo);
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));;

    // ---- Without preprocessing - set blob
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", modelToUse.width, modelToUse.height, IE::NHWC, true);
    IE::PreProcessInfo preprocInfoDefault;
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob, preprocInfoDefault));

    // ---- Without preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    outputBlob = inferRequest.GetBlob(outputBlobName);
    refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);
    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(toFP32(outputBlob), toFP32(refBlob), numberOfTopClassesToCompare));
}
