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

#include "blob_descriptor.h"

#include <gtest/gtest.h>

#include "creators/creator_blob.h"
using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

class BlobDescriptor_UnitTests : public ::testing::Test, public ::testing::WithParamInterface<BlobDescType> {
public:
    BlobDescType blobType = BlobDescType::ImageWorkload;
    IE::DataPtr inputDesc =
            std::make_shared<IE::Data>("input", IE::Precision::U8, IE::Layout::NCHW);
    IE::DataPtr outputDesc =
            std::make_shared<IE::Data>("output", IE::Precision::U8, IE::Layout::NCHW);
    IE::Blob::Ptr blob = Blob_Creator::createBlob({1, 1, 1, 1}, IE::Layout::NCHW);

public:
    void SetUp() {
        blobType = GetParam();
    }
};

//------------------------------------------------------------------------------
TEST_P(BlobDescriptor_UnitTests, input_NoThrow) {
    EXPECT_NO_THROW(BlobDescriptorAdapter blobDescriptor(blobType, inputDesc, blob));
}

TEST_P(BlobDescriptor_UnitTests, output_NoThrow) {
    EXPECT_NO_THROW(BlobDescriptorAdapter blobDescriptor(blobType, outputDesc, nullptr));
}

TEST_P(BlobDescriptor_UnitTests, monkey_NullDesc) {
    EXPECT_ANY_THROW(BlobDescriptorAdapter blobDescriptor(blobType, nullptr, blob));
}


//------------------------------------------------------------------------------
const static std::vector<BlobDescType> blobTypes = {BlobDescType::VideoWorkload, BlobDescType::ImageWorkload};

INSTANTIATE_TEST_CASE_P(BlobDescType, BlobDescriptor_UnitTests, ::testing::ValuesIn(blobTypes));
