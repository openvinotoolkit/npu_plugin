//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

namespace {

//
// inferOutputLayout
//

mlir::FailureOr<DimsOrder> inferOutputLayout(const DimArr& inPerm, mlir::ArrayAttr dimMapAttr) {
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimMapAttr);
    SmallVector<vpux::Dim> perm;

    // Iterate over input dims in the given order and push back corresponding output dims as indicated by the op's
    // dim_mapping. The result is the permutation of output dims.
    bool layoutInferFail = false;
    for (auto pIt = inPerm.begin(); pIt != inPerm.end(); ++pIt) {
        const auto outputDims = dimMapping[pIt->ind()];
        for (const auto& dim : outputDims) {
            const auto outDim = vpux::Dim(dim);

            // Ensure input dim order is not switched.
            // E.g. nchw -> c'h'w', with n = c', c = h', h * w = w'
            // Layouts 0123 and 0132 would both produce 012 output layout, but
            // the content of w' would not be the same.
            if (!perm.empty() && perm.back() == outDim) {
                layoutInferFail = std::prev(pIt)->ind() > pIt->ind();
                if (layoutInferFail == true) {
                    return mlir::failure();
                }

                continue;
            }
            perm.push_back(outDim);
        }
    }

    // Check that the resulting output permutation does not have duplicate dims
    SmallVector<vpux::Dim> temp(perm);
    llvm::sort(temp.begin(), temp.end(), [](const vpux::Dim& dim0, const vpux::Dim& dim1) {
        return dim0.ind() < dim1.ind();
    });

    if (std::adjacent_find(temp.begin(), temp.end()) != temp.end())
        return mlir::failure();
    return DimsOrder::fromPermutation(makeArrayRef(perm));
}

mlir::FailureOr<mlir::Type> inferElemType(VPU::AffineReshapeOpAdaptor affineReshapeOp, mlir::Type inputElemType) {
    const auto perAxisQType = inputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
    if (perAxisQType == nullptr) {
        return inputElemType;
    }

    const auto inputQAxis = perAxisQType.getQuantizedDimension();

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.dim_mapping());
    const auto outputShape = parseIntArrayAttr<int64_t>(affineReshapeOp.shape_value());
    const auto inputShape = getShape(affineReshapeOp.input()).raw();

    // get output dims for input Q axis
    const auto outputDims = dimMapping[inputQAxis];
    int64_t outQAxis = -1;
    int64_t inputQAxisSize = inputShape[inputQAxis];

    if (inputQAxisSize == 1) {
        // Per tensor, but must be per channel, do not handle it here
        return mlir::failure();
    }

    for (const auto& dim : outputDims) {
        if (inputQAxisSize == outputShape[dim]) {
            // firstly check that element is unique and others == 1
            if (std::find_if(outputDims.begin(), outputDims.end(), [&](int64_t elem) {
                    return (outputShape[elem] != 1 && outputShape[elem] != inputQAxisSize);
                }) != outputDims.end()) {
                return mlir::failure();
            }
            outQAxis = dim;
            break;
        }
    }

    if (outQAxis == -1) {
        return mlir::failure();
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), static_cast<int32_t>(outQAxis),
            perAxisQType.getStorageTypeMin(), perAxisQType.getStorageTypeMax());
}

}  // namespace

mlir::LogicalResult vpux::VPU::AffineReshapeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::AffineReshapeOpAdaptor affineReshape(operands, attrs);
    if (mlir::failed(affineReshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = Shape(parseIntArrayAttr<int64_t>(affineReshape.shape_value()));
    const auto input = affineReshape.input();
    const auto inType = input.getType();
    const auto ndInType = inType.cast<vpux::NDTypeInterface>();
    const auto inOrder = DimsOrder::fromValue(input);

    const auto outputLayout = inferOutputLayout(inOrder.toPermutation(), affineReshape.dim_mapping());
    if (mlir::failed(outputLayout)) {
        return mlir::failure();
    }

    const auto typeComponents = TypeComponents().setShape(outShape).setDimsOrder(outputLayout.getValue());
    auto outType = ndInType.changeTypeComponents(typeComponents);

    const auto elemTypeInferResult = inferElemType(affineReshape, ndInType.getElementType());
    if (mlir::succeeded(elemTypeInferResult)) {
        outType = outType.changeElemType(elemTypeInferResult.getValue());
    }
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::VPU::AffineReshapeOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    auto outputElemType = inferElemType(*this, info.getInput(0));
    if (mlir::failed(outputElemType)) {
        return;
    }

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, outputElemType.getValue());
    }
}

void vpux::VPU::AffineReshapeOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
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
// inferLayoutInfo
//

void vpux::VPU::AffineReshapeOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto inOrder = info.getInput(0);
    const auto inPermutation = inOrder.toPermutation();
    const auto outPermutation = inferOutputLayout(inPermutation, dim_mapping());
    if (mlir::failed(outPermutation)) {
        IE::fillDefaultLayoutInfo(info);
        return;
    }

    info.setInput(0, inOrder);
    info.setOutput(0, outPermutation.getValue());
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::AffineReshapeOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReshapeParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReshapeParams});
}
