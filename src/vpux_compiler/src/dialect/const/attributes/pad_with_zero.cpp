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
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// PadWithZeroAttr::verify
//

mlir::LogicalResult vpux::Const::PadWithZeroAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                         mlir::ArrayAttr padBefore, mlir::ArrayAttr padAfter) {
    if (padBefore == nullptr) {
        return printTo(emitError(), "Got NULL 'padBefore' in 'PadWithZeroAttr'");
    }
    if (padAfter == nullptr) {
        return printTo(emitError(), "Got NULL 'padAfter' in 'PadWithZeroAttr'");
    }

    if (padBefore.size() != padAfter.size()) {
        return printTo(emitError(), "Got non consistent 'padBefore' and 'padAfter' values in 'PadWithZeroAttr'");
    }

    for (const auto dimAttr : padBefore.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value in 'padBefore' for 'PadWithZeroAttr'");
        }
    }
    for (const auto dimAttr : padAfter.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value in 'padAfter' for 'PadWithZeroAttr'");
        }
    }

    return mlir::success();
}

//
// PadWithZeroAttr::print
//

void vpux::Const::PadWithZeroAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getPadBefore());
    printer << ", ";
    printer.printAttribute(getPadAfter());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::PadWithZeroAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr padBefore;
    if (mlir::failed(parser.parseAttribute(padBefore))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr padAfter;
    if (mlir::failed(parser.parseAttribute(padAfter))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::PadWithZeroAttr>(padBefore, padAfter);
}

//
// PadWithZeroAttr::inferOutputType
//

mlir::ShapedType vpux::Const::PadWithZeroAttr::inferOutputType(mlir::ShapedType input) const {
    const auto inShape = getShape(input);
    const auto padBefore = Shape(parseIntArrayAttr(getPadBefore()));
    const auto padAfter = Shape(parseIntArrayAttr(getPadAfter()));

    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(),
                      "Got non consistent 'padBefore' and 'padAfter' values in 'PadWithZeroAttr'");
    VPUX_THROW_UNLESS(inShape.size() == padBefore.size(),
                      "Paddings and input shape are not consistent in 'PadWithZeroAttr'");

    Shape outShape(inShape.size());
    for (auto ind : irange(outShape.size())) {
        const auto d = Dim(ind);
        outShape[d] = inShape[d] + padBefore[d] + padAfter[d];
    }

    return input.clone(outShape.raw());
}

//
// PadWithZeroAttr::transform
//

Const::Content vpux::Const::PadWithZeroAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(), false);

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();

    std::fill_n(outBuf.data(), outBuf.size(), char(0));

    const auto padBefore = Shape(parseIntArrayAttr(getPadBefore()));

    const Byte elemSize = getElemTypeSize(input.getStorageElemType());
    const auto order = DimsOrder::fromType(input.getType());

    const auto inShape = getShape(input.getType());
    const auto inMemShape = order.toMemoryOrder(inShape);

    const auto outShape = getShape(output.getType());
    const auto outMemShape = order.toMemoryOrder(outShape);

    loop_1d(LoopExecPolicy::Parallel, input.getNumElements(), [&](int64_t inMemInd1D) {
        const auto inMemIndND = getMemIndexND(inMemInd1D, inMemShape);
        const auto inIndND = order.toLogicalOrder(inMemIndND);

        Shape outIndND(inIndND.size());
        for (auto ind : irange(outIndND.size())) {
            const auto d = Dim(ind);
            outIndND[d] = inIndND[d] + padBefore[d];
        }

        const auto outMemIndND = order.toMemoryOrder(outIndND);
        const auto outMemInd1D = getMemIndex1D(outMemIndND, outMemShape);

        const auto inMemRawInd = input.isSplat() ? 0 : checked_cast<size_t>(inMemInd1D * elemSize.count());
        VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'PadWithZeroAttr'");

        const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
        VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'PadWithZeroAttr'");

        std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()), outBuf.data() + outMemRawInd);
    });

    return output;
}

//
// ContentAttr::padWithZero
//

Const::ContentAttr vpux::Const::ContentAttr::padWithZero(ShapeRef padBefore, ShapeRef padAfter) const {
    return get(*this, Const::PadWithZeroAttr::get(getInt64ArrayAttr(getContext(), padBefore),
                                                  getInt64ArrayAttr(getContext(), padAfter))
                              .cast<Const::TransformAttrInterface>());
}
