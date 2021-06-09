//
// Copyright 2020-2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
