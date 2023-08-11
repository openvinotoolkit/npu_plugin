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

mlir::FailureOr<SmallVector<int64_t>> getAxes(VPU::SqueezeOpAdaptor squeeze, mlir::Location loc) {
    if (squeeze.axes() != nullptr && squeeze.axes_value().hasValue()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (squeeze.axes() == nullptr && !squeeze.axes_value().hasValue()) {
        return errorAt(loc, "Missed axes representation");
    }

    if (squeeze.axes_value().hasValue()) {
        return parseIntArrayAttr<int64_t>(squeeze.axes_value().getValue());
    }

    auto axesConst = squeeze.axes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.content();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());
    std::sort(axes.begin(), axes.end());

    const auto inType = squeeze.input().getType().cast<vpux::NDTypeInterface>();
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    return axes;
}

//
// inferOutputLayout
//

DimsOrder inferOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axesVec, ArrayRef<int64_t> inShape) {
    SmallVector<vpux::Dim> perm;
    SmallVector<int64_t> axes = axesVec;

    // If axes attr is empty, find all dims equal to 1
    if (axes.empty()) {
        for (auto inInd : irange(inShape.size())) {
            if (inShape[inInd] == 1) {
                axes.push_back(inInd);
            }
        }
    }

    // Iterate over input dims in the given order and push back corresponding output dims.
    for (const auto& p : inPerm) {
        // Skip over squeezed dim
        if (llvm::find(axes, p.ind()) != axes.end())
            continue;

        auto dim = p.ind();
        // Decrement input dim index by the number of squeezed axes smaller than itself
        for (const auto& squeezeAxis : axes) {
            if (p.ind() > squeezeAxis) {
                dim--;
            }
        }

        perm.push_back(vpux::Dim(dim));
    }

    return DimsOrder::fromPermutation(makeArrayRef(perm));
}

}  // namespace

mlir::LogicalResult vpux::VPU::SqueezeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                           mlir::Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SqueezeOpAdaptor squeeze(operands, attrs);
    if (mlir::failed(squeeze.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = getAxes(squeeze, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto input = squeeze.input();
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto inOrder = DimsOrder::fromValue(input);

    SmallVector<int64_t> outShape;

    if (axes->empty()) {
        for (auto dim : inShape) {
            if (dim != 1) {
                outShape.push_back(dim);
            }
        }
    } else {
        size_t axesInd = 0;
        for (auto inInd : irange(inShape.size())) {
            if (axesInd < axes->size()) {
                const auto nextAxisInd = checked_cast<size_t>(axes.getValue()[axesInd]);

                if (nextAxisInd < inInd) {
                    return errorAt(loc, "Axis '{0}' was occurred twice", nextAxisInd);
                }

                if (nextAxisInd == inInd) {
                    if (inShape[inInd] != 1) {
                        return errorAt(loc, "Can't exclude '{0}' dimension, it is not equal to 1", nextAxisInd);
                    }

                    ++axesInd;

                    continue;
                }
            }

            outShape.push_back(inShape[inInd]);
        }
    }

    const auto typeComponents =
            TypeComponents()
                    .setShape(Shape(outShape))
                    .setDimsOrder(inferOutputLayout(inOrder.toPermutation(), axes.getValue(), inShape));
    auto outType = inType.changeTypeComponents(typeComponents);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// inferLayoutInfo
//

void vpux::VPU::SqueezeOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto axes = parseIntArrayAttr<int64_t>(axes_value().getValue());
    const auto inShape = input().getType().cast<mlir::RankedTensorType>().getShape();
    const auto inOrder = info.getInput(0);
    const auto inPermutation = inOrder.toPermutation();

    info.setInput(0, inOrder);
    info.setOutput(0, inferOutputLayout(inPermutation, axes, inShape));
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::SqueezeOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReshapeParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReshapeParams});
}
