#include "vpux/compiler/utils/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

namespace vpux {
namespace utils {

mlir::Value alignDepthwiseWeightTensor(mlir::OpBuilder& builder, mlir::Location loc, const mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[IE::Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[IE::Dims4D::Filter::IC];
    const auto KY = filterShape[IE::Dims4D::Filter::KY];
    const auto KX = filterShape[IE::Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto depthwiseConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % depthwiseConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);
    if (remainder == 0) {
        // nothing to align
        return origFilter;
    }

    auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(weightsConst != nullptr, "Grouped convolution does not provide constant weights");

    const int64_t alignment = depthwiseConvAlignment - remainder;
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightShape = Shape{OC, filtersPerInChan * KY * KX, 1, 1};
    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, alignment, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    auto alignedWeightShape = SmallVector<int64_t>{OC, filtersPerInChan * KY * KX + alignment, 1, 1};
    const auto outAllocType = mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType());
    const auto outAllocTypeNHWC = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

}  // namespace utils
}  // namespace vpux
