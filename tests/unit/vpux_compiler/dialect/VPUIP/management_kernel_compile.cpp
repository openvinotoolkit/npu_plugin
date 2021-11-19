//
// Copyright Intel Corporation.
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

#include <vpux/compiler/act_kernels/compilation.h>

#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(ManagementKernel, Compile) {
    if (!checkVpuip2Dir()) {
        GTEST_SKIP() << "Skip due to VPUIP_2_Directory environment variable isn't set";
    }

    const auto params = vpux::VPUIP::BlobWriter::compileParams();

    ActKernelDesc desc;

    EXPECT_NO_THROW(desc = compileManagementKernelForACTShave(params));
}
