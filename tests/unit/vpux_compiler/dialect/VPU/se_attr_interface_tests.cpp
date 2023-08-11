//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux/compiler/core/attributes/shape.hpp>
#include <vpux/compiler/core/attributes/strides.hpp>
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <gtest/gtest.h>

using namespace vpux;

struct SEInterpolateAttrParams {
    VPU::NCEInterpolateMode mode;
    VPU::NCEInterpolateNearestMode nearestMode;
    VPU::NCEInterpolateCoordMode coordMode;
    std::vector<double> scales;
    std::vector<int64_t> offsets;
    std::vector<int64_t> sizes;
    std::vector<int64_t> dataShape;
    std::vector<Bit> dataStrides;
    int64_t dataElemByteSize;
    int64_t seSize;
    std::vector<int32_t> expectedOutput;
};

class SEInterpolateAttrTests : public testing::TestWithParam<SEInterpolateAttrParams> {};

TEST_P(SEInterpolateAttrTests, ComputeSEOffsets) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto params = GetParam();

    auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, params.mode);
    auto nearestModeAttr = VPU::NCEInterpolateNearestModeAttr::get(&ctx, params.nearestMode);
    auto coordModeAttr = VPU::NCEInterpolateCoordModeAttr::get(&ctx, params.coordMode);
    auto scaleAttr = getFPArrayAttr(&ctx, params.scales);
    auto offsetsAttr = params.offsets.empty() ? nullptr : getIntArrayAttr(&ctx, params.offsets);
    auto sizesAttr = params.sizes.empty() ? nullptr : getIntArrayAttr(&ctx, params.sizes);
    auto interpolateAttr = VPU::SEInterpolateAttr::get(&ctx, modeAttr, nearestModeAttr, coordModeAttr, scaleAttr,
                                                       offsetsAttr, sizesAttr);

    auto seAttrInterface = interpolateAttr.dyn_cast<VPU::SEAttr>();
    ASSERT_TRUE(seAttrInterface != nullptr);

    Shape dataShape(params.dataShape);
    Strides dataStrides(params.dataStrides);
    Byte elemSize(params.dataElemByteSize);
    const auto seOffsets = seAttrInterface.computeSEOffsets(dataShape, dataStrides, elemSize, params.seSize);
    EXPECT_EQ(seOffsets, params.expectedOutput);
}

// clang-format off

std::vector<SEInterpolateAttrParams> nearestAsymmetricParams = {
    //
    // Nearest modes
    //
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::ROUND_PREFER_FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128, \
                         96, 96, 112, 112, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::ROUND_PREFER_CEIL, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32,  32,  32, \
                         48,  64,  64,  80,  80,  80, \
                         48,  64,  64,  80,  80,  80, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128, \
                         96, 96, 112, 112, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::CEIL, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32,  32,  32, \
                         48,  64,  64,  80,  80,  80, \
                         48,  64,  64,  80,  80,  80, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::SIMPLE, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128, \
                         96, 96, 112, 112, 128, 128}},

    //
    // Scales
    //
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 3, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  16,  16,  16,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 4, 4}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128}},

    //
    // Element byte sizes
    //
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/2, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0,  32,  32,  64,  64, \
                           0,   0,  32,  32,  64,  64, \
                          96,  96, 128, 128, 160, 160, \
                          96,  96, 128, 128, 160, 160, \
                         192, 192, 224, 224, 256, 256, \
                         192, 192, 224, 224, 256, 256}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/4, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0, 64,   64, 128, 128, \
                           0,   0, 64,   64, 128, 128, \
                         192, 192, 256, 256, 320, 320, \
                         192, 192, 256, 256, 320, 320, \
                         384, 384, 448, 448, 512, 512, \
                         384, 384, 448, 448, 512, 512}},

    //
    // Storage element sizes
    //
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 32, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{  0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80, \
                           0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 64, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, \
                           0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 96, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, \
                           0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832}},

    //
    // Offsets & sizes
    //
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 0, 0}, /*sizes=*/{1, 16, 5, 6},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 1, 1}, /*sizes=*/{1, 16, 4, 4},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32, \
                         48,  64,  64,  80, \
                         48,  64,  64,  80, \
                         96, 112, 112, 128}},
};

std::vector<SEInterpolateAttrParams> bilinearAsymmetricParams = {
    //
    // Scales
    //
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32,  32, \
                          0,  0,  16,  16,  32,  32,  32, \
                         48, 48,  64,  64,  80,  80,  80, \
                         48, 48,  64,  64,  80,  80,  80, \
                         96, 96, 112, 112, 128, 128, 128, \
                         96, 96, 112, 112, 128, 128, 128, \
                         96, 96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 3, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  16,  16,  16,  32,  32,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32,  32,  32, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80,  80,  80, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128}},
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 4, 4}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128}},

    //
    // Element byte sizes
    //
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/2, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0,  32,  32,  64,  64, 64, \
                           0,   0,  32,  32,  64,  64, 64, \
                          96,  96, 128, 128, 160, 160, 160, \
                          96,  96, 128, 128, 160, 160, 160, \
                         192, 192, 224, 224, 256, 256, 256, \
                         192, 192, 224, 224, 256, 256, 256, \
                         192, 192, 224, 224, 256, 256, 256}},
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/4, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0, 64,   64, 128, 128, 128, \
                           0,   0, 64,   64, 128, 128, 128, \
                         192, 192, 256, 256, 320, 320, 320, \
                         192, 192, 256, 256, 320, 320, 320, \
                         384, 384, 448, 448, 512, 512, 512, \
                         384, 384, 448, 448, 512, 512, 512, \
                         384, 384, 448, 448, 512, 512, 512}},

    //
    // Storage element sizes
    //
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 32, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{  0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80,  64,  80, \
                           0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80,  64,  80, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, 160, 176, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, 160, 176, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, 256, 272, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, 256, 272, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, 256, 272}},
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 64, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, 128, 160, \
                           0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, 128, 160, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, 320, 352, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, 320, 352, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, 512, 544, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, 512, 544, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, 512, 544}},
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 96, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, 192, 224, 256, \
                           0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, 192, 224, 256, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, 480, 512, 544, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, 480, 512, 544, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, 768, 800, 832, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, 768, 800, 832, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, 768, 800, 832}},

    //
    // Offsets & sizes
    //
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 0, 0}, /*sizes=*/{1, 16, 6, 7},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32,  32, \
                          0,  0,  16,  16,  32,  32,  32, \
                         48, 48,  64,  64,  80,  80,  80, \
                         48, 48,  64,  64,  80,  80,  80, \
                         96, 96, 112, 112, 128, 128, 128, \
                         96, 96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::BILINEAR, VPU::NCEInterpolateNearestMode::FLOOR, VPU::NCEInterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 1, 1}, /*sizes=*/{1, 16, 5, 5},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32,  32, \
                         48,  64,  64,  80,  80, \
                         48,  64,  64,  80,  80, \
                         96, 112, 112, 128, 128, \
                         96, 112, 112, 128, 128}},
};

// clang-format on

INSTANTIATE_TEST_CASE_P(NearestAsymmetric, SEInterpolateAttrTests, testing::ValuesIn(nearestAsymmetricParams));
INSTANTIATE_TEST_CASE_P(BilinearAsymmetric, SEInterpolateAttrTests, testing::ValuesIn(bilinearAsymmetricParams));
