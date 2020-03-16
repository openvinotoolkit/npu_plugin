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
class BlobDescriptor_UnitTests : public ::testing::Test {
public:
    InferenceEngine::DataPtr desc = std::make_shared<IE::Data>("input", IE::Precision::U8, IE::Layout::NCHW);
    InferenceEngine::Blob::Ptr blob = Blob_Creator::createBlob({1, 1, 1, 1}, IE::Layout::NCHW);
};

//------------------------------------------------------------------------------
using LocalBlobDescriptor_constructor = BlobDescriptor_UnitTests;
TEST_F(LocalBlobDescriptor_constructor, default_NoThrow) {
    EXPECT_NO_THROW(LocalBlobDescriptor blobDescriptor(desc, blob));
}

TEST_F(LocalBlobDescriptor_constructor, monkey_NullDesc) {
    EXPECT_ANY_THROW(LocalBlobDescriptor blobDescriptor(nullptr, blob));
}

TEST_F(LocalBlobDescriptor_constructor, monkey_NullBlob) {
    EXPECT_ANY_THROW(LocalBlobDescriptor blobDescriptor(desc, nullptr));
}

//------------------------------------------------------------------------------
using RemoteBlobDescriptor_constructor = BlobDescriptor_UnitTests;
TEST_F(RemoteBlobDescriptor_constructor, default_NoThrow) {
    EXPECT_NO_THROW(RemoteBlobDescriptor blobDescriptor(desc, blob));
}

TEST_F(RemoteBlobDescriptor_constructor, monkey_NullDesc) {
    EXPECT_ANY_THROW(RemoteBlobDescriptor blobDescriptor(nullptr, blob));
}

TEST_F(RemoteBlobDescriptor_constructor, monkey_NullBlob) {
    EXPECT_ANY_THROW(RemoteBlobDescriptor blobDescriptor(desc, nullptr));
}
