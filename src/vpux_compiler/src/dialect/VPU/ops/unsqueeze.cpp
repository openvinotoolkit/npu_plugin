//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// getAxes
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getAxes(VPU::UnsqueezeOpAdaptor unsqueeze, mlir::Location loc) {
    if (unsqueeze.axes() != nullptr && unsqueeze.axes_value() != nullptr) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (unsqueeze.axes() == nullptr && unsqueeze.axes_value() == nullptr) {
        return errorAt(loc, "Missed axes representation");
    }

    if (unsqueeze.axes_value() != nullptr) {
        return parseIntArrayAttr<int64_t>(unsqueeze.axes_value());
    }

    auto axesConst = unsqueeze.axes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.content();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());
    std::sort(axes.begin(), axes.end());

    const auto inType = unsqueeze.input().getType().cast<vpux::NDTypeInterface>();
    const auto inRank = inType.getRank();
    const auto numAxes = checked_cast<int64_t>(axes.size());

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank + numAxes;
        }
    }

    return axes;
}

//
// inferOutputLayout
//

DimsOrder inferOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axes) {
    SmallVector<vpux::Dim> perm;

    // Iterate over input dims in the given order and push back corresponding output dims.
    for (const auto& p : inPerm) {
        auto dim = p.ind();
        for (const auto& unsqueezedAxis : axes) {
            if (dim > unsqueezedAxis) {
                dim++;
            } else if (dim == unsqueezedAxis) {
                perm.push_back(vpux::Dim(dim));
                dim++;
            }
        }

        perm.push_back(vpux::Dim(dim));
    }

    // If unsqueezed 1s are at the end, push their corresponding axes in the perm vec
    const auto sz = static_cast<int64_t>(perm.size());
    for (const auto& unsqueezedAxis : axes) {
        if (unsqueezedAxis >= sz) {
            perm.push_back(vpux::Dim(unsqueezedAxis));
        }
    }

    return DimsOrder::fromPermutation(makeArrayRef(perm));
}

}  // namespace

mlir::LogicalResult vpux::VPU::UnsqueezeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             mlir::Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::UnsqueezeOpAdaptor unsqueeze(operands, attrs);
    if (mlir::failed(unsqueeze.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = getAxes(unsqueeze, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto input = unsqueeze.input();
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto inOrder = DimsOrder::fromValue(input);

    SmallVector<int64_t> outShape(inShape.size() + axes->size());

    size_t inInd = 0;
    size_t axesInd = 0;
    for (auto outInd : irange(outShape.size())) {
        if (axesInd < axes.getValue().size()) {
            const auto nextAxisInd = checked_cast<size_t>(axes.getValue()[axesInd]);

            if (nextAxisInd < outInd) {
                return errorAt(loc, "Axis '{0}' was occurred twice", nextAxisInd);
            }

            if (nextAxisInd == outInd) {
                outShape[outInd] = 1;
                ++axesInd;
                continue;
            }
        }

        if (inInd < inShape.size()) {
            outShape[outInd] = inShape[inInd];
            ++inInd;
            continue;
        }
    }
    if (inInd != inShape.size() || axesInd != axes->size()) {
        return errorAt(loc, "Inconsistent parameters");
    }

    auto outType = inType.changeDimsOrder(inferOutputLayout(inOrder.toPermutation(), axes.getValue()));
    outType = outType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// inferLayoutInfo
//

void vpux::VPU::UnsqueezeOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto axes = parseIntArrayAttr<int64_t>(axes_value().getValue());
    const auto inOrder = info.getInput(0);
    const auto inPermutation = inOrder.toPermutation();

    info.setInput(0, inOrder);
    info.setOutput(0, inferOutputLayout(inPermutation, axes));
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::UnsqueezeOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReshapeParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReshapeParams});
}
