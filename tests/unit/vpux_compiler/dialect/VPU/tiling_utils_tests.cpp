//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"

#include <gtest/gtest.h>
#include <mlir/Parser.h>

using namespace vpux;

TEST(MLIR_VPU_TilingUtils, BackInferPadsTile) {
    const auto compareInferredPads = [&](ShapeRef inputShape, PadInfo padInfo, ArrayRef<int64_t> kernelSize,
                                         ArrayRef<int64_t> kernelStrides, ShapeRef tileShape, ShapeRef tileOffsets,
                                         PadInfo expectedPads) {
        TileInfo outTile(tileShape);
        outTile.offsets = Shape(tileOffsets.raw());
        outTile.axis[Dims4D::Act::H] = 5;
        const auto inferredPads = backInferPadsTile(outTile, inputShape, padInfo, kernelSize, kernelStrides);
        EXPECT_EQ(inferredPads, expectedPads);
    };

    {
        const Shape inShape{1, 16, 7, 7};
        const PadInfo padInfo{0, 0, 0, 0};
        const SmallVector<int64_t> kernelSize{1, 1};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{0, 0, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{0, 0, 0, 0});
    }

    {
        const Shape inShape{1, 16, 9, 9};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{0, 0, 0, 0};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{0, 0, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{0, 0, 0, 0});
    }

    {
        const Shape inShape{1, 16, 7, 7};
        const PadInfo padInfo{1, 1, 1, 1};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{1, 1, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{1, 1, 0, 1});
    }

    {
        const Shape inShape{1, 16, 13, 13};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{1, 1, 1, 1};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{2, 2};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{1, 1, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{1, 1, 0, 1});
    }

    {
        const Shape inShape{1, 16, 7, 7};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{2, 2, 2, 2};
        const SmallVector<int64_t> kernelSize{5, 5};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 2, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{2, 2, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 2, 0}, /*expectedPads=*/{2, 2, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 2, 0, 1});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{2, 2, 0, 2});

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 2, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 2, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 2, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 2, 0, 2});
    }

    {
        const Shape inShape{1, 16, 14, 14};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{2, 2, 2, 2};
        const SmallVector<int64_t> kernelSize{5, 5};
        const SmallVector<int64_t> kernelStrides{2, 2};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 1, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{2, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{2, 1, 0, 1});
    }
}
