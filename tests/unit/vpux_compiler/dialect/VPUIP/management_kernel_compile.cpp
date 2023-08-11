//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/act_kernels/compilation.h>

#include <gtest/gtest.h>

using namespace vpux;

TEST(ManagementKernel, Compile) {
    ActKernelDesc desc;

#if defined(_WIN32) || defined(_WIN64)  // Skipped temporary on Windows (E#26870)
    GTEST_SKIP() << "Skip Windows validation";
#endif
    EXPECT_NO_THROW(desc = compileManagementKernelForACTShave());
}
