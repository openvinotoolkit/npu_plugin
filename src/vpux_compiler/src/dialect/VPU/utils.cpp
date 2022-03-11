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

#include "vpux/compiler/dialect/VPU/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

//
// DW Convolution utility
//

namespace {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   Shape flatWeightShape, int64_t padding) {
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, padding, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto origFilterType = weightsConst.output().getType().cast<vpux::NDTypeInterface>();
    const auto outAllocType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                      .cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

Const::ContentAttr buildPadData(const mlir::Type type, ArrayRef<int64_t> shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "Unsupported shape size {0}", shape.size());
    const auto OC = shape[Dims4D::Filter::OC.ind()];
    const auto alignment = shape[Dims4D::Filter::IC.ind()];

    if (const auto quantizedType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto padType = mlir::RankedTensorType::get(shape, normalizeQuantStorageType(quantizedType));
        std::vector<int64_t> padValues;

        if (const auto uniformType = quantizedType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
            padValues.assign(OC * alignment, uniformType.getZeroPoint());
        } else if (const auto perAxisType = quantizedType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto zeroPoints = perAxisType.getZeroPoints();
            VPUX_THROW_UNLESS(checked_cast<size_t>(OC) == zeroPoints.size(),
                              "Number of zero points {0} and channels {1} don't match", zeroPoints.size(), OC);

            // assuming all zero points are equal to broadcast
            VPUX_THROW_UNLESS(
                    zeroPoints.size() == 1 || std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin()),
                    "All zero points should be equal");
            padValues.assign(OC * alignment, zeroPoints.front());

        } else {
            VPUX_THROW("Unsupported Quantized Type '{0}'", quantizedType);
        }
        std::vector<uint8_t> padValuesUint8;
        std::transform(padValues.begin(), padValues.end(), std::back_inserter(padValuesUint8),
                       [](int64_t value) -> uint8_t {
                           return static_cast<uint8_t>(value);
                       });
        const auto padAttr = mlir::DenseElementsAttr::get(padType, makeArrayRef(padValuesUint8));

        return Const::ContentAttr::get(padAttr).quantCast(quantizedType);
    } else {
        const auto ndType = mlir::RankedTensorType::get(shape, type).cast<vpux::NDTypeInterface>();
        const auto padType = ndType.changeDimsOrder(DimsOrder::NCHW).cast<mlir::RankedTensorType>();
        const auto padValues = std::vector<ngraph::float16>(OC * alignment, 0.f);
        const auto padAttr = mlir::DenseElementsAttr::get(padType, makeArrayRef(padValues));

        return Const::ContentAttr::get(padAttr);
    }
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      Shape flatWeightShape, int64_t padding) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    //    const auto origOrder = origFilterType.getDimsOrder();
    const auto flatWeightType = origFilterType.changeShape(flatWeightShape).changeDimsOrder(DimsOrder::NCHW);
    std::cout << llvm::formatv("flatWeightShape {0}", flatWeightShape).str() << std::endl;
    auto flatWeightsOp = builder.create<IE::ReshapeOp>(loc, flatWeightType, origFilter, nullptr, false,
                                                       getIntArrayAttr(ctx, flatWeightShape));
    flatWeightsOp->dump();

    // Step 2: Permute flat input to NCHW.
    //    auto flatWeightTypeNCHWType = flatWeightType.changeDimsOrder(DimsOrder::NCHW);
    //    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    //    const auto flatWeightsDimsAttr =
    //            mlir::AffineMapAttr::get(getPermutationFromOrders(origOrder, DimsOrder::NCHW, ctx));
    //    auto flatWeightsNCHW = builder.create<IE::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp.output(),
    //                                                             nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto outShapedType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                       .cast<vpux::NDTypeInterface>();
    const auto outAllocType = outShapedType.changeDimsOrder(DimsOrder::NHWC);

    const auto padShape = SmallVector<int64_t>{OC, padding, 1, 1};
    const auto padContentAttr = buildPadData(origFilterType.getElementType(), padShape);

    const auto padAllocType =
            mlir::RankedTensorType::get(padShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    ;
    const auto padAllocTypeNHWC = padAllocType.changeDimsOrder(DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, padContentAttr);

    // Step 4: Concatenate flat NCHW input with padding.

    auto concatViewOp =
            builder.create<IE::ConcatOp>(loc, SmallVector<mlir::Value>{flatWeightsOp, paddedTensor}, Dims4D::Act::C);

    // Step 5: Permute the result to NHWC.
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(DimsOrder::NCHW, DimsOrder::NHWC, ctx));

    auto outOpNCHW = builder.create<IE::PermuteCastOp>(loc, outAllocType, concatViewOp.output(), nhwcAttr, memPermAttr);

    return outOpNCHW.output();
}

}  // namespace

mlir::Value vpux::VPU::alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                   mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto padding = alignment - remainder;

    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};

    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        return getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, padding);
    } else {
        return getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, padding);
    }
}

//
// CM Convolution utility
//

mlir::Value vpux::VPU::alignChannelMajorWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                      mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (IC * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto padding = alignment - remainder;

    auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(weightsConst != nullptr, "Channel Major convolution does not provide constant weights");

    const auto weightsContentAttr = weightsConst.contentAttr();

    const auto flatWeightShape = Shape{OC, 1, 1, IC * KY * KX};
    const auto flatWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW).reshape(flatWeightShape);
    const auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, 0, 0, padding});
    const auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
    const auto outAllocType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                      .cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);

    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);
    return alignedWeightsOp.output();
}
