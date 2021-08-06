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
#include "vpux/compiler/utils/types.hpp"

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
    const auto padBefore = Shape(parseIntArrayAttr<int64_t>(getPadBefore()));
    const auto padAfter = Shape(parseIntArrayAttr<int64_t>(getPadAfter()));

    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(),
                      "Got non consistent 'padBefore' and 'padAfter' values in 'PadWithZeroAttr'");
    VPUX_THROW_UNLESS(inShape.size() == padBefore.size(),
                      "Paddings and input shape are not consistent in 'PadWithZeroAttr'");

    Shape outShape(inShape.size());
    for (auto ind : irange(outShape.size())) {
        const auto d = Dim(ind);
        outShape[d] = inShape[d] + padBefore[d] + padAfter[d];
    }

    return changeShape(input, outShape);
}

//
// PadWithZeroAttr::transform
//

Const::Content vpux::Const::PadWithZeroAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(), false);

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();

    std::fill_n(outBuf.data(), outBuf.size(), char(0));

    const Byte elemSize = getElemTypeSize(input.getStorageElemType());
    const auto order = DimsOrder::fromType(input.getType());

    const auto inShape = getShape(input.getType());
    const auto inMemShape = order.toMemoryOrder(inShape);

    const auto outShape = getShape(output.getType());
    const auto outMemShape = order.toMemoryOrder(outShape);

    const auto padBefore = Shape(parseIntArrayAttr<int64_t>(getPadBefore()));
    const auto memPadBefore = order.toMemoryOrder(padBefore);

    if (memPadBefore.size() == 1) {
        // Opitimized 1D case

        if (input.isSplat()) {
            const auto md0 = MemDim(0);
            const auto IN0 = inMemShape[md0];
            const auto off0 = memPadBefore[md0];

            loop_1d(LoopExecPolicy::Parallel, IN0, [&](int64_t in0) {
                const auto out0 = in0 + off0;

                std::copy_n(inBuf.data(), checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + checked_cast<size_t>(out0 * elemSize.count()));
            });
        } else {
            std::copy_n(inBuf.data(), checked_cast<size_t>(input.getNumElements() * elemSize.count()),
                        outBuf.data() + memPadBefore.front() * elemSize.count());
        }
    } else if (memPadBefore.size() == 2) {
        // Opitimized 2D case

        const auto md0 = MemDim(0);
        const auto md1 = MemDim(1);

        const auto IN0 = inMemShape[md0];
        const auto IN1 = inMemShape[md1];

        const auto OUT1 = outMemShape[md1];

        const auto off0 = memPadBefore[md0];
        const auto off1 = memPadBefore[md1];

        loop_2d(LoopExecPolicy::Parallel, IN0, IN1, [&](int64_t in0, int64_t in1) {
            const auto out0 = in0 + off0;
            const auto out1 = in1 + off1;

            const auto outRawInd = out1 + out0 * OUT1;
            const auto inRawInd = input.isSplat() ? 0 : in1 + in0 * IN1;

            std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                        checked_cast<size_t>(elemSize.count()),
                        outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
        });
    } else if (memPadBefore.size() == 3) {
        // Opitimized 3D case

        const auto md0 = MemDim(0);
        const auto md1 = MemDim(1);
        const auto md2 = MemDim(2);

        const auto IN0 = inMemShape[md0];
        const auto IN1 = inMemShape[md1];
        const auto IN2 = inMemShape[md2];

        const auto OUT1 = outMemShape[md1];
        const auto OUT2 = outMemShape[md2];

        const auto off0 = memPadBefore[md0];
        const auto off1 = memPadBefore[md1];
        const auto off2 = memPadBefore[md2];

        loop_3d(LoopExecPolicy::Parallel, IN0, IN1, IN2, [&](int64_t in0, int64_t in1, int64_t in2) {
            const auto out0 = in0 + off0;
            const auto out1 = in1 + off1;
            const auto out2 = in2 + off2;

            const auto outRawInd = out2 + out1 * OUT2 + out0 * OUT2 * OUT1;
            const auto inRawInd = input.isSplat() ? 0 : in2 + in1 * IN2 + in0 * IN2 * IN1;

            std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                        checked_cast<size_t>(elemSize.count()),
                        outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
        });
    } else if (memPadBefore.size() == 4) {
        // Opitimized 4D case

        const auto md0 = MemDim(0);
        const auto md1 = MemDim(1);
        const auto md2 = MemDim(2);
        const auto md3 = MemDim(3);

        const auto IN0 = inMemShape[md0];
        const auto IN1 = inMemShape[md1];
        const auto IN2 = inMemShape[md2];
        const auto IN3 = inMemShape[md3];

        const auto OUT1 = outMemShape[md1];
        const auto OUT2 = outMemShape[md2];
        const auto OUT3 = outMemShape[md3];

        const auto off0 = memPadBefore[md0];
        const auto off1 = memPadBefore[md1];
        const auto off2 = memPadBefore[md2];
        const auto off3 = memPadBefore[md3];

        loop_4d(LoopExecPolicy::Parallel, IN0, IN1, IN2, IN3, [&](int64_t in0, int64_t in1, int64_t in2, int64_t in3) {
            const auto out0 = in0 + off0;
            const auto out1 = in1 + off1;
            const auto out2 = in2 + off2;
            const auto out3 = in3 + off3;

            const auto outRawInd = out3 + out2 * OUT3 + out1 * OUT3 * OUT2 + out0 * OUT3 * OUT2 * OUT1;
            const auto inRawInd = input.isSplat() ? 0 : in3 + in2 * IN3 + in1 * IN3 * IN2 + in0 * IN3 * IN2 * IN1;

            std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                        checked_cast<size_t>(elemSize.count()),
                        outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
        });
    } else {
        // Generic case

        loop_1d(LoopExecPolicy::Parallel, input.getNumElements(), [&](int64_t inMemInd1D) {
            const auto inMemIndND = getMemIndexND(inMemInd1D, inMemShape);

            MemShape outMemIndND(inMemIndND.size());
            for (auto ind : irange(outMemIndND.size())) {
                const auto md = MemDim(ind);
                outMemIndND[md] = inMemIndND[md] + memPadBefore[md];
            }

            const auto outMemInd1D = getMemIndex1D(outMemIndND, outMemShape);

            const auto inMemRawInd = input.isSplat() ? 0 : checked_cast<size_t>(inMemInd1D * elemSize.count());
            VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'PadWithZeroAttr'");

            const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
            VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'PadWithZeroAttr'");

            std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                        outBuf.data() + outMemRawInd);
        });
    }

    return output;
}

//
// ContentAttr::padWithZero
//

Const::ContentAttr vpux::Const::ContentAttr::padWithZero(ShapeRef padBefore, ShapeRef padAfter) const {
    return get(*this, Const::PadWithZeroAttr::get(getIntArrayAttr(getContext(), padBefore),
                                                  getIntArrayAttr(getContext(), padAfter))
                              .cast<Const::TransformAttrInterface>());
}
