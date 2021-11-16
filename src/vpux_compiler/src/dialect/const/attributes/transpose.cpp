//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/hash.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <unordered_map>
#include <unordered_set>

using namespace vpux;

//
// TransposeAttr::walkImmediateSubElements
//

void vpux::Const::TransposeAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                          llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getOrder());
}

//
// TransposeAttr::verify
//

mlir::LogicalResult vpux::Const::TransposeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                       mlir::AffineMapAttr order) {
    if (order == nullptr) {
        return printTo(emitError(), "Got NULL 'order' in 'TransposeAttr'");
    }

    if (!order.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'order' in 'TransposeAttr'");
    }

    return mlir::success();
}

//
// TransposeAttr::print
//

void vpux::Const::TransposeAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getOrder());
    printer << ">";
}

//
// TransposeAttr::parse
//

mlir::Attribute vpux::Const::TransposeAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::AffineMapAttr order;
    if (mlir::failed(parser.parseAttribute(order))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::TransposeAttr>(order);
}

//
// TransposeAttr::inferOutputType
//

mlir::ShapedType vpux::Const::TransposeAttr::inferOutputType(mlir::ShapedType input) const {
    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);

    const auto inputShape = getShape(input);
    SmallVector<Dim> perm(order.numDims());
    for (size_t idx = 0; idx < perm.size(); idx++) {
        perm[idx] = order.dimAt(idx);
    }

    Shape newShape(inputShape.size());
    for (size_t idx = 0; idx < newShape.size(); idx++) {
        newShape[Dim(idx)] = inputShape[perm[idx]];
    }

    return changeShape(input, newShape);
}

//
// TransposeAttr::transform
//

Const::Content vpux::Const::TransposeAttr::transform(vpux::Const::Content& input) const {
    // FIXME this whole part was borrowed from reorder transformation as is.
    // Find a way to re-use this code.
    auto output = Const::Content::allocTempBuffer(input.getType(), input.getStorageElemType(), input.isSplat());

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();
    VPUX_THROW_UNLESS(outBuf.size() == inBuf.size(), "Storage buffer size mismatch in 'TransposeAttr'");

    const auto inOrder = DimsOrder::fromType(input.getType());
    const auto outOrder = DimsOrder::fromAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(inOrder.numDims() == outOrder.numDims(), "Can't transpose from '{0}' to '{1}'", inOrder,
                      outOrder);

    if (input.isSplat() || inOrder == outOrder) {
        std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());
    } else {
        const Byte elemSize = getElemTypeSize(input.getStorageElemType());
        const auto shape = getShape(output.getType());

        static const std::unordered_set<std::pair<DimsOrder, DimsOrder>> optimizedCases = {
                {DimsOrder::NCHW, DimsOrder::NHWC},
                {DimsOrder::NHWC, DimsOrder::NCHW},
                {DimsOrder::NCDHW, DimsOrder::NDHWC},
                {DimsOrder::NDHWC, DimsOrder::NCDHW},
        };

        static const std::unordered_map<size_t, InferenceEngine::Precision> elemSizeToPrecision = {
                {sizeof(uint8_t), InferenceEngine::Precision::U8},
                {sizeof(uint16_t), InferenceEngine::Precision::U16},
                {sizeof(uint32_t), InferenceEngine::Precision::U32},
                // U64 is not supported by cvtBlobLayout
                // {sizeof(uint64_t), InferenceEngine::Precision::U64},
        };

        const auto precision = elemSizeToPrecision.find(checked_cast<size_t>(elemSize.count()));

        if (optimizedCases.count({inOrder, outOrder}) != 0 && precision != elemSizeToPrecision.end()) {
            // Use optimized algorithm from IE core

            const InferenceEngine::SizeVector ieShape(shape.begin(), shape.end());

            const InferenceEngine::TensorDesc inDesc(precision->second, ieShape, inOrder.toIE());
            const InferenceEngine::TensorDesc outDesc(precision->second, ieShape, outOrder.toIE());

            const auto inBlob = makeBlob(inDesc, nullptr, const_cast<char*>(inBuf.data()));
            const auto outBlob = makeBlob(outDesc, nullptr, outBuf.data());

            cvtBlobLayout(inBlob, outBlob);
        } else {
            // Use generic algorithm

            const auto inMemShape = inOrder.toMemoryOrder(shape);
            const auto outMemShape = outOrder.toMemoryOrder(shape);

            loop_1d(LoopExecPolicy::Parallel, output.getType().getNumElements(), [&](int64_t outMemInd1D) {
                const auto outMemIndND = getMemIndexND(outMemInd1D, outMemShape);
                const auto indND = outOrder.toLogicalOrder(outMemIndND);
                const auto inMemIndND = inOrder.toMemoryOrder(indND);
                const auto inMemInd1D = getMemIndex1D(inMemIndND, inMemShape);

                const auto inMemRawInd = checked_cast<size_t>(inMemInd1D * elemSize.count());
                VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'TransposeAttr'");

                const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
                VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'TransposeAttr'");

                std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + outMemRawInd);
            });
        }
    }

    auto transposedOutput = Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                            input.getStorageElemType(), input.isSplat());
    auto transposedOutputBuf = transposedOutput.getRawTempBuf();
    std::copy_n(outBuf.data(), outBuf.size(), transposedOutputBuf.data());

    return transposedOutput;
}

//
// ContentAttr::transpose
//

Const::ContentAttr vpux::Const::ContentAttr::transpose(DimsOrder newOrder) const {
    return get(*this, Const::TransposeAttr::get(mlir::AffineMapAttr::get(newOrder.toAffineMap(getContext())))
                              .cast<Const::TransformAttrInterface>());
}
