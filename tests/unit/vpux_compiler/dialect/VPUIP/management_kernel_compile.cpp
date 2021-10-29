
#include "vpux/compiler/act_kernels/act_kernel_gen.h"

#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(ManagementKernel, Compile) {
    if (!checkVpuip2Dir()) {
        GTEST_SKIP() << "Skip due to VPUIP_2_Directory environment variable isn't set";
    }

    const auto params = vpux::VPUIP::BlobWriter::compileParams();

    flatbuffers::FlatBufferBuilder fbb;
    ActKernelDesc desc;

    EXPECT_NO_THROW(desc = compileManagementKernelForACTShave(params, fbb));
}
