//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <vpux_encryption.h>

#if defined(__arm__) || defined(__aarch64__)

using VPUXEncryptUnitTests = ::testing::Test;


TEST_F(VPUXEncryptUnitTests, canCreateEncryptionNoThrow) {
    ASSERT_NO_THROW(vpux::Encryption encryptionModel);
    vpux::Encryption encryptionModel;
    ASSERT_NO_THROW(encryptionModel.isLibraryFound());
}

TEST_F(VPUXEncryptUnitTests, canCreateEncryptionWithFakeLibPathNoThrow) {
    vpux::Encryption encryptionModel("fake_lib_name.so");
    ASSERT_NO_THROW(encryptionModel.isLibraryFound());
}

#endif
