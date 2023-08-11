//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";

}  // namespace

TEST(MLIR_MemoryContiguousWithTilingTest, SegmentedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    // Segmented on H, #NHWC
    {
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto numClustersAttr = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr,
                                                numClustersAttr, nullptr, nullptr, nullptr, nullptr, nullptr, &ctx);

        const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
        const auto elemType = mlir::Float16Type::get(&ctx);

        const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
        const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr,
                                                   /*allocSize=*/nullptr, &ctx);

        const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        EXPECT_EQ(VPUIP::isMemoryContiguousWithTiling(distributedBufferType), true);
    }

    // Segmented on C, #NHWC
    {
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
        const auto numClustersAttr = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr,
                                                numClustersAttr, nullptr, nullptr, nullptr, nullptr, nullptr, &ctx);

        const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
        const auto elemType = mlir::Float16Type::get(&ctx);

        const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
        const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr,
                                                   /*allocSize=*/nullptr, &ctx);

        const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        EXPECT_EQ(VPUIP::isMemoryContiguousWithTiling(distributedBufferType), false);
    }

    // Segmented on C, #NCHW
    {
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
        const auto numClustersAttr = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr,
                                                numClustersAttr, nullptr, nullptr, nullptr, nullptr, nullptr, &ctx);

        const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
        const auto elemType = mlir::Float16Type::get(&ctx);

        const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(&ctx));
        const auto elemStrides = SmallVector<int64_t>({64 * 13 * 16, 13 * 16, 16, 1});
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr,
                                                   /*allocSize=*/nullptr, &ctx);

        const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

        const auto distributedBufferType =
                VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

        EXPECT_EQ(VPUIP::isMemoryContiguousWithTiling(distributedBufferType), true);
    }
}

TEST(MLIR_MemoryContiguousWithTilingTest, OverlappedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 2);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr,
                                               /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // SOH overlapped, 1x1s1p0
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto pads = VPU::PaddingAttr::get(getIntAttr(&ctx, 0), getIntAttr(&ctx, 0), getIntAttr(&ctx, 0),
                                            getIntAttr(&ctx, 0), &ctx);
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto distributedAttr =
            VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, &ctx);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    EXPECT_EQ(VPUIP::isMemoryContiguousWithTiling(distributedBufferType), true);
}

TEST(MLIR_MemoryContiguousWithTilingTest, SegmentedDuplicatedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 2);
    const auto distributedAttr =
            VPU::DistributedTensorAttr::get(distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr,
                                            numClustersAttr, nullptr, nullptr, nullptr, nullptr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr,
                                               /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);
    EXPECT_EQ(VPUIP::isMemoryContiguousWithTiling(distributedBufferType), true);
}

TEST(MLIR_MemoryContiguousWithTilingTest, DuplicatedDistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
    const auto numClustersAttr = getIntAttr(&ctx, 2);
    const auto distributedAttr =
            VPU::DistributedTensorAttr::get(distributionModeAttr, nullptr, nullptr, nullptr, nullptr, numClustersAttr,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, &ctx);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr,
                                               /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);
    EXPECT_EQ(VPUIP::isMemoryContiguousWithTiling(distributedBufferType), true);
}
