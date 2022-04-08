//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

namespace {

mlir::SmallVector<int64_t> calcTileOutputShape(mlir::Value input, Const::DeclareOp repeats) {
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();

    const auto repeatsContent = repeats.content();
    const auto repeatsVals = repeatsContent.getValues<int64_t>();

    auto outShape = to_small_vector(inType.getShape());

    // If number of elements in *"repeats"* is more than shape of *"data"*, then *"data"* will be promoted to
    // "*repeats*" by prepending new axes, e.g. let's shape of *"data"* is equal to (2, 3) and *"repeats"* is equal to
    // [2, 2, 2], then shape of *"data"* will be promoted to (1, 2, 3) and result shape will be (2, 4, 6).
    //
    // If number of elements in *"repeats"* is less than shape of *"data"*, then *"repeats"* will be promoted to
    // "*data*" by prepending 1's to it, e.g. let's shape of *"data"* is equal to (4, 2, 3) and *"repeats"* is equal to
    // [2, 2], then *"repeats"* will be promoted to [1, 2, 2] and result shape will be (4, 4, 6)

    while (repeatsVals.size() > outShape.size()) {
        outShape.insert(outShape.begin(), 1);
    }

    auto out_shape_iter = std::prev(outShape.end());
    auto repeats_iter = std::prev(repeatsVals.end());
    for (; out_shape_iter != std::prev(outShape.begin()) && repeats_iter != std::prev(repeatsVals.begin());
         --out_shape_iter, --repeats_iter) {
        *out_shape_iter *= *repeats_iter;
    }
    return outShape;
}

}  // namespace

mlir::LogicalResult vpux::VPU::TileOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::TileOpAdaptor tile(operands, attrs);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = tile.input().getType().cast<vpux::NDTypeInterface>();
    auto repeatsConst = tile.repeats().getDefiningOp<Const::DeclareOp>();
    if (repeatsConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for repeats");
    }

    auto outShape = calcTileOutputShape(tile.input(), repeatsConst);

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::TileOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("Unreacheable code, since all tile ops are converted to VPU::PerAxisTileOp");
}
