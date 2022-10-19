//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DepthToSpaceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::DepthToSpaceOpAdaptor depthToSpace(operands, attrs);
    if (mlir::failed(depthToSpace.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(depthToSpace.input());
    const auto inType = depthToSpace.input().getType().cast<vpux::NDTypeInterface>();
    const auto block_size = depthToSpace.block_size().getInt();

    const auto elemType = inType.getElementType();
    if (!(elemType.isF16() || elemType.isF32() || elemType.isUnsignedInteger(8))) {
        return errorAt(loc, "DepthToSpace only support FP16, FP32, U8 data type");
    }

    if (inShape.size() < 3) {
        return errorAt(loc, "Invalid input tensor shape, dimension must be greater than 2.");
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    if (inShape[Dims4D::Act::C] % (block_size * block_size) != 0) {
        return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                       inShape[Dims4D::Act::C]);
    }

    size_t W_out = inShape[Dims4D::Act::W] * block_size;
    size_t H_out = inShape[Dims4D::Act::H] * block_size;
    size_t C_out = inShape[Dims4D::Act::C] / (block_size * block_size);
    size_t N_out = inShape[Dims4D::Act::N];

    SmallVector<int64_t> outShape{checked_cast<int64_t>(N_out), checked_cast<int64_t>(C_out),
                                  checked_cast<int64_t>(H_out), checked_cast<int64_t>(W_out)};

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::DepthToSpaceOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::DepthToSpaceParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(block_size());
    builder.add_blockSize(blockSize);

    builder.add_mode(vpux::VPUIP::convertVPUXDepthToSpaceMode2MVCNN(mode()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DepthToSpaceParams});
}
