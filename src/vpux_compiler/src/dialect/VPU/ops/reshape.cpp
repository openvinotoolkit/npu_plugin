//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// getOutShape
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getOutShape(VPU::ReshapeOpAdaptor reshape, mlir::Location loc) {
    if (reshape.shape() != nullptr && reshape.shape_value().hasValue()) {
        return errorAt(loc, "Ambiguous shape representation");
    }
    if (reshape.shape() == nullptr && !reshape.shape_value().hasValue()) {
        return errorAt(loc, "Missed shape representation");
    }

    if (reshape.shape_value().hasValue()) {
        return parseIntArrayAttr<int64_t>(reshape.shape_value().getValue());
    }

    auto shapeConst = reshape.shape().getDefiningOp<Const::DeclareOp>();
    if (shapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for shape");
    }

    const auto shapeContent = shapeConst.content();
    auto shapeVec = to_small_vector(shapeContent.getValues<int64_t>());

    const auto specialZero = reshape.special_zero();

    const auto zeroDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == 0;
    });
    const auto negativeDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == -1;
    });

    if (negativeDims > 1) {
        return errorAt(loc, "Shape can not contain more than 1 negative value");
    }

    if (!(zeroDims != 0 && specialZero) && negativeDims == 0) {
        return shapeVec;
    } else {
        const auto inShape = to_small_vector(reshape.input().getType().cast<vpux::NDTypeInterface>().getShape().raw());

        auto dividend = std::accumulate(inShape.begin(), inShape.end(), int64_t(1), std::multiplies<int64_t>());

        for (size_t i = 0; i < shapeVec.size(); ++i) {
            auto& v = shapeVec[i];

            if (v == 0 && specialZero) {
                if (i >= inShape.size()) {
                    return errorAt(loc, "Shape value at '{0}' is out of range '{1}'", i, inShape.size());
                }

                v = inShape[i];
            }

            if (v > 0) {
                if (dividend % v != 0) {
                    return errorAt(loc, "Shape value at '{0}' ('{1}') is invalid", i, v);
                }

                dividend /= v;
            }
        }

        if (negativeDims > 0) {
            const auto negIt = std::find(shapeVec.begin(), shapeVec.end(), -1);
            VPUX_THROW_UNLESS(negIt != shapeVec.end(), "Shape vector broken");

            *negIt = dividend;
        }

        return shapeVec;
    }
}

}  // namespace

mlir::LogicalResult vpux::VPU::ReshapeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                           mlir::Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ReshapeOpAdaptor reshape(operands, attrs);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(reshape, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto inType = reshape.input().getType().cast<vpux::NDTypeInterface>();

    const auto typeComponents = TypeComponents()
                                        .setShape(Shape(outShape.getValue()))
                                        .setDimsOrder(DimsOrder::fromNumDims(outShape->size()));
    auto outType = inType.changeTypeComponents(typeComponents);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::VPU::ReshapeOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto inputElemType = info.getInput(0);

    // Do not propagate element type down in per channel case.
    // E#31030
    if (inputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() == nullptr) {
        for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
            info.setOutput(outputInd, inputElemType);
        }
    }
}

void vpux::VPU::ReshapeOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (outputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() != nullptr) {
        // E#31029: implement propagate type up for per channel, currently it leads to failures in later passes.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReshapeOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReshapeParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReshapeParams});
}
