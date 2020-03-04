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

#include <blob_factory.hpp>
#include <ie_core.hpp>

#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "ie_blob.h"
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

class ImageWorkload_Tests : public ::testing::Test {
public:
    std::string graphPath;
    std::string refInputPath;
    std::string refOutputPath;

    const size_t numberOfTopClassesToCompare = 5;

protected:
    void SetUp() override;
    static void printRawBlob(const IE::Blob::Ptr& blob, const size_t& sizeToPrint, const std::string& blobName = "");
};

void ImageWorkload_Tests::SetUp() {
    graphPath = PrecompiledResNet_Helper::resnet50_dpu.graphPath;
    refInputPath = PrecompiledResNet_Helper::resnet50_dpu.inputPath;
    refOutputPath = PrecompiledResNet_Helper::resnet50_dpu.outputPath;
}

void ImageWorkload_Tests::printRawBlob(
    const IE::Blob::Ptr& blob, const size_t& sizeToPrint, const std::string& blobName) {
    if (blob->size() < sizeToPrint) {
        THROW_IE_EXCEPTION << "Blob size is smaller then required size to print.\n"
                              "Blob size: "
                           << blob->size() << " sizeToPrint: " << sizeToPrint;
    }
    if (blob->getTensorDesc().getPrecision() != IE::Precision::U8) {
        THROW_IE_EXCEPTION << "Unsupported precision";
    }

    auto mBlob = IE::as<IE::MemoryBlob>(blob);
    auto mappedMemory = mBlob->rmap();
    uint8_t* data = mappedMemory.as<uint8_t*>();
    std::cout << "uint8_t output raw data" << (blobName.empty() ? "" : " for " + blobName) << "\t: " << std::hex;
    for (size_t i = 0; i < sizeToPrint; ++i) {
        std::cout << unsigned(data[i]) << " ";
    }
    std::cout << std::endl;
}

TEST_F(ImageWorkload_Tests, SyncInference) {
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
