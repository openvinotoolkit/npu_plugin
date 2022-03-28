//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux/compiler/act_kernels/compilation.h>

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(ManagementKernel, Compile) {

    const auto params = vpux::VPUIP::BlobWriter::compileParams();

    ActKernelDesc desc;

#if defined(_WIN32) || defined(_WIN64)  // Skipped temporary on Windows (EISW-26870)
    GTEST_SKIP() << "Skip Windows validation";
#endif
    EXPECT_NO_THROW(desc = compileManagementKernelForACTShave(params));
}
