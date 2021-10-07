//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

namespace vpux {
namespace VPUIP {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   Shape flatWeightShape, const int64_t alignment) {
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, alignment, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + alignment, 1, 1};
    const auto origFilterType = weightsConst.output().getType().cast<mlir::ShapedType>();
    const auto outAllocType = mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType());
    const auto outAllocTypeNHWC = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      Shape flatWeightShape, const int64_t alignment) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto flatWeightType =
            changeDimsOrder(changeShape(origFilterType, flatWeightShape), DimsOrder::fromValue(origFilter));
    auto flatWeightsOp = builder.create<IERT::GenericReshapeOp>(loc, flatWeightType, origFilter);

    // Step 2: Permute flat input to NCHW.
    auto flatWeightTypeNCHWType = changeDimsOrder(flatWeightType, DimsOrder::NCHW);
    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto flatWeightsDimsAttr =
            mlir::AffineMapAttr::get(DimsOrder::fromValue(flatWeightsOp.output()).toAffineMap(ctx));
    auto flatWeightsNCHW = builder.create<IERT::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp.output(),
                                                               nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + alignment, 1, 1};
    const auto outAllocType = changeDimsOrder(
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()), DimsOrder::NCHW);

    const auto padShape = SmallVector<int64_t>{OC, alignment, 1, 1};
    const auto padValues = std::vector<ngraph::float16>(OC * alignment, 0.f);
    const auto padType =
            changeDimsOrder(mlir::RankedTensorType::get(padShape, origFilterType.getElementType()), DimsOrder::NCHW);
    const auto padAttr = mlir::DenseElementsAttr::get(padType, makeArrayRef(padValues));
    const auto padContentAttr = Const::ContentAttr::get(padAttr);

    const auto padAllocType = mlir::MemRefType::get(padShape, origFilterType.getElementType());
    const auto padAllocTypeNHWC = changeDimsOrder(padAllocType, DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, padContentAttr);

    // Step 4: Concatenate flat NCHW input with padding.
    auto subViewAlloc = builder.create<mlir::memref::AllocOp>(loc, outAllocType);

    const SmallVector<int64_t> filterOffsets = {0, 0, 0, 0};
    const auto filterOffsetsAttr = getIntArrayAttr(ctx, filterOffsets);
    const auto flatWeightShapeAttr = getIntArrayAttr(ctx, flatWeightShape);

    const SmallVector<int64_t> paddingOffsets = {0, flatWeightChannelsCount, 0, 0};
    const auto paddingOffsetsAttr = getIntArrayAttr(ctx, paddingOffsets);
    const auto padShapeAttr = getIntArrayAttr(ctx, padShape);

    auto subViewFilter = builder.create<IERT::SubViewOp>(loc, subViewAlloc, filterOffsetsAttr, flatWeightShapeAttr);
    auto subViewPadding = builder.create<IERT::SubViewOp>(loc, subViewAlloc, paddingOffsetsAttr, padShapeAttr);

    auto subViewFilterCopy = builder.create<IERT::CopyOp>(loc, flatWeightsNCHW.result(), subViewFilter);
    auto subViewPaddingCopy = builder.create<IERT::CopyOp>(loc, paddedTensor.output(), subViewPadding);

    auto concatViewOp = builder.create<IERT::ConcatViewOp>(
            loc, SmallVector<mlir::Value>{subViewFilterCopy.output(), subViewPaddingCopy.output()}, subViewAlloc);

    // Step 5: Permute the result to NHWC.
    auto outNHWCType = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));

    auto outOpNCHW = builder.create<IERT::PermuteCastOp>(loc, outNHWCType, concatViewOp.output(), nhwcAttr, nchwAttr);

    return outOpNCHW.result();
}

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, const mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto depthwiseConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % depthwiseConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);
    if (remainder == 0) {
        // nothing to align
        return origFilter;
    }

    const int64_t alignment = depthwiseConvAlignment - remainder;
    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};
    mlir::Value alignedFilter;
    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        alignedFilter = getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, alignment);
    } else {
        alignedFilter = getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, alignment);
    }
    return alignedFilter;
}

unsigned int swizzle_addr(unsigned int addr, unsigned char key) {
    // Some constant get automatically optimized by the compiler
    const unsigned int LOG2_RAM_CUT_BYTES = 9;                // address size of 32 KB RAM cut with 128 bits words
    const unsigned int CUT_ADDRESS_MASK_10b = (1 << 11) - 1;  // RAM cut address mask
    const unsigned int MAX_SWIZZLE_KEY = 5;
    const unsigned int RAM_CUT_ADDRESS_MASK = (1 << LOG2_RAM_CUT_BYTES) - 1;

    unsigned int stagger_address_mask = (1 << key) - 1;
    int shift = LOG2_RAM_CUT_BYTES - key;
    unsigned int addr_stagger, phy_addr;

    addr_stagger = (addr >> 4) & CUT_ADDRESS_MASK_10b;   // get the address in the ramcut
    addr_stagger = addr_stagger >> MAX_SWIZZLE_KEY;      // right shift 5 bits
    addr_stagger = addr_stagger & stagger_address_mask;  // get only the relevant bits of the address
    addr_stagger = addr_stagger << shift;                // Shift them back the their original locaiton

    phy_addr = addr + addr_stagger;
    phy_addr = phy_addr & RAM_CUT_ADDRESS_MASK;
    phy_addr = phy_addr + (addr & (~RAM_CUT_ADDRESS_MASK));

    return phy_addr;
}

template <class T>
void swizzled_img(T* input_array, T* output_array, const unsigned char key, const unsigned int size) {
    for (unsigned int addr = 0; addr < size; addr++) {
        output_array[swizzle_addr(addr, key)] = input_array[addr];
    }
}

template <class T>
int swizzled_size(T*, const unsigned char key, const unsigned int size) {
    unsigned int max_size = 0;
    for (unsigned int addr = 0; addr < size; addr++) {
        max_size = std::max(max_size, swizzle_addr(addr, key));
    }
    // Align to 64 bytes
    return (((max_size + 63) / 64) * 64);
}

}  // namespace VPUIP
}  // namespace vpux
