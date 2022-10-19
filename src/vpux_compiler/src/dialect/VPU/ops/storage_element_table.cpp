//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

void vpux::VPU::StorageElementTableOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                             int64_t seDepth, int64_t seSize, int64_t height, int64_t width,
                                             int32_t base_ptr) {
    auto numPtrs = seDepth * width * height;
    SmallVector<int32_t> basePtrs(numPtrs, base_ptr);
    build(odsBuilder, odsState, seDepth, seSize, height, width, getIntArrayAttr(odsBuilder, basePtrs));
}

mlir::LogicalResult vpux::VPU::StorageElementTableOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::StorageElementTableOpAdaptor setOp(operands, attrs);
    if (mlir::failed(setOp.verify(loc))) {
        return mlir::failure();
    }

    const auto depth = setOp.seDepth();
    const auto width = setOp.width();
    const auto height = setOp.height();

    SmallVector<mlir::IntegerAttr> dims{depth, height, width};

    auto shape = to_small_vector(dims | transformed([](mlir::IntegerAttr attr) {
                                     return checked_cast<int64_t>(attr.getValue().getSExtValue());
                                 }));
    shape.insert(shape.begin(), 1);
    const auto outType = getTensorType(ShapeRef(shape), getUInt32Type(ctx), DimsOrder::NCHW, nullptr);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::verifyOp(vpux::VPU::StorageElementTableOp setOp) {
    using namespace VPU::NCESparsity;

    const auto basePtrs = parseIntArrayAttr<int32_t>(setOp.base_ptrs());
    const auto expectedNumPtrs = static_cast<size_t>(setOp.seDepth() * setOp.width() * setOp.height());
    if (expectedNumPtrs != basePtrs.size()) {
        return errorAt(setOp->getLoc(), "StorageElementTable expects to have {0}, but got {1}", expectedNumPtrs,
                       basePtrs.size());
    }

    const mlir::DenseSet<int32_t> distinctPtrs(basePtrs.begin(), basePtrs.end());
    if (distinctPtrs.size() > MAX_DISTINCT_BASE_PTRS) {
        return errorAt(
                setOp->getLoc(),
                "StorageElementTable expects to have at most {0} unique values in the base_ptr array, but have {1}.",
                MAX_DISTINCT_BASE_PTRS, distinctPtrs.size());
    }

    const auto allPtrsAreValid = llvm::all_of(basePtrs, [](auto ptr) {
        return 0 <= ptr && ptr <= MAX_BASE_POINTER_VALUE;
    });
    if (!allPtrsAreValid) {
        return errorAt(setOp->getLoc(), "Operation contains invalid base_ptrs");
    }

    return mlir::success();
}
