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
// ReorderAttr::verify
//

mlir::LogicalResult vpux::Const::ReorderAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::AffineMapAttr order) {
    if (order == nullptr) {
        return printTo(emitError(), "Got NULL 'order' in 'ReorderAttr'");
    }

    if (!order.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'order' in 'ReorderAttr'");
    }

    return mlir::success();
}

//
// ReorderAttr::print
//

void vpux::Const::ReorderAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getOrder());
    printer << ">";
}

//
// ReorderAttr::parse
//

mlir::Attribute vpux::Const::ReorderAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser, mlir::Type) {
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

    return parser.getChecked<Const::ReorderAttr>(order);
}

//
// ReorderAttr::inferOutputType
//

mlir::ShapedType vpux::Const::ReorderAttr::inferOutputType(mlir::ShapedType input) const {
    const auto order = DimsOrder::fromPermutationAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);

    return changeDimsOrder(input, order);
}

//
// ReorderAttr::transform
//

Const::Content vpux::Const::ReorderAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(),
                                                  input.isSplat());

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();
    VPUX_THROW_UNLESS(outBuf.size() == inBuf.size(), "Storage buffer size mismatch in 'ReorderAttr'");

    const auto inOrder = DimsOrder::fromType(input.getType());
    const auto outOrder = DimsOrder::fromType(output.getType());
    VPUX_THROW_UNLESS(inOrder.numDims() == outOrder.numDims(), "Can't reorder from '{0}' to '{1}'", inOrder, outOrder);

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
                VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'ReorderAttr'");

                const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
                VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'ReorderAttr'");

                std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + outMemRawInd);
            });
        }
    }

    return output;
}

//
// ContentAttr::reorder
//

Const::ContentAttr vpux::Const::ContentAttr::reorder(DimsOrder newOrder) const {
    return get(*this, Const::ReorderAttr::get(mlir::AffineMapAttr::get(newOrder.toPermutationAffineMap(getContext())))
                              .cast<Const::TransformAttrInterface>());
}
