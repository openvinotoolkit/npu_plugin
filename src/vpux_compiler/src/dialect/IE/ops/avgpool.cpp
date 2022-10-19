//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::AvgPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AvgPoolOpAdaptor avgPool(operands, attrs);
    if (mlir::failed(avgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(avgPool.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(avgPool.pads_begin());
    const auto windowShape = parseIntArrayAttr<int64_t>(avgPool.kernel_size());
    const auto windowStrides = parseIntArrayAttr<int64_t>(avgPool.strides());
    const auto roundingType = avgPool.rounding_type().getValue();

    const auto inType = avgPool.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = avgPool.input().getType().cast<mlir::ShapedType>().getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()),
            true, /* It is only used during assertion. True will make it pass */
            roundingType == vpux::IE::RoundingType::CEIL);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

//
// inferLayoutInfo
//

void vpux::IE::AvgPoolOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto arch = VPU::getArch((*this)->getParentOfType<mlir::ModuleOp>());
    if (arch == VPU::ArchKind::VPUX37XX) {
        const auto logCb = [&](const formatv_object_base&) {};
        if (!VPU::NCEAveragePoolOp::isSupported(*this, logCb, false, false)) {
            // Operation will be done on software so dims order have no restriction.
            info.setOutput(0, info.getInput(0));
        } else {
            info.setInput(0, DimsOrder::NHWC);
            info.setOutput(0, DimsOrder::NHWC);
        }
    } else {
        info.setOutput(0, info.getInput(0));
    }
}

//
// verifyChannels
//

mlir::LogicalResult vpux::IE::AvgPoolOp::verifyChannels() {
    const auto arch = VPU::getArch((*this)->getParentOfType<mlir::ModuleOp>());
    if (arch == VPU::ArchKind::VPUX37XX) {
        const auto logCb = [&](const formatv_object_base&) {};
        if (!VPU::NCEAveragePoolOp::isSupported(*this, logCb, false, false)) {
            // Operation will be done on software so that channels have no restriction
            return mlir::success();
        }

        const auto inputType = (*this).input().getType().cast<vpux::NDTypeInterface>();
        if (inputType.getRank() != 4) {
            return mlir::failure();
        }

        const auto inputShape = inputType.getShape();
        const auto IC = inputShape[Dims4D::Act::C];

        if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

int64_t vpux::IE::AvgPoolOp::getInputChannelAlignment() {
    const auto arch = VPU::getArch((*this)->getParentOfType<mlir::ModuleOp>());
    if (arch == VPU::ArchKind::VPUX37XX) {
        const auto inputType = input().getType().cast<vpux::NDTypeInterface>();
        return VPU::NCEInvariant::getAlignment(inputType.getElementType());
    }

    return 1;
}

int64_t vpux::IE::AvgPoolOp::getOutputChannelAlignment() {
    const auto arch = VPU::getArch((*this)->getParentOfType<mlir::ModuleOp>());
    if (arch == VPU::ArchKind::VPUX37XX) {
        const auto outputType = output().getType().cast<vpux::NDTypeInterface>();
        return VPU::NCEInvariant::getAlignment(outputType.getElementType());
    }

    return 1;
}
