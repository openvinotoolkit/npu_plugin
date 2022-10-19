//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SpaceToDepthOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::SpaceToDepthOpAdaptor spd(operands, attrs);
    if (mlir::failed(spd.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = spd.input().getType().cast<vpux::NDTypeInterface>();

    const auto elementType = inputType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isUnsignedInteger(8))) {
        return errorAt(loc, "SpaceToDepth only supports FP16, FP32, U8 input data type");
    }

    const auto inputShape = inputType.getShape().raw();
    const auto block_size = spd.block_size().getInt();

    if (inputShape.size() < 3) {
        return errorAt(loc, "Input tensor rank must be greater than 2. Got {0}D tensor", inputShape.size());
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (inputShape[H.ind()] % block_size || inputShape[W.ind()] % block_size) {
        return errorAt(loc, "Invalid block_size {0} , height {1} and width {2} must be divisible by block_size",
                       block_size, inputShape[H.ind()], inputShape[W.ind()]);
    }

    const auto outN = inputShape[N.ind()];
    const auto outC = inputShape[C.ind()] * block_size * block_size;
    const auto outH = inputShape[H.ind()] / block_size;
    const auto outW = inputShape[W.ind()] / block_size;

    SmallVector<int64_t> outShape{outN, outC, outH, outW};

    const auto outType = inputType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::SpaceToDepthOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::SpaceToDepthParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(block_size());
    builder.add_blockSize(blockSize);

    const auto spdMode = VPUIP::convertVPUXSpaceToDepthMode2MVCNN(mode());
    builder.add_mode(spdMode);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SpaceToDepthParams});
}
