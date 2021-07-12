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
// SubViewAttr::verify
//

mlir::LogicalResult vpux::Const::SubViewAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ArrayAttr offset, mlir::ArrayAttr shape) {
    if (offset == nullptr) {
        return printTo(emitError(), "Got NULL 'offset' in 'SubViewAttr'");
    }
    if (shape == nullptr) {
        return printTo(emitError(), "Got NULL 'shape' in 'SubViewAttr'");
    }

    if (offset.size() != shape.size()) {
        return printTo(emitError(), "Got inconsistent 'offset' and 'shape' values in 'SubViewAttr'");
    }

    for (const auto dimAttr : offset.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'offset' for 'SubViewAttr'", dimAttr);
        }
        if (dimAttr.cast<mlir::IntegerAttr>().getInt() < 0) {
            return printTo(emitError(), "Got unsupported dimension value '{0}' in 'offset' for 'SubViewAttr'", dimAttr);
        }
    }

    for (const auto dimAttr : shape.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'shape' for 'SubViewAttr'", dimAttr);
        }
        if (dimAttr.cast<mlir::IntegerAttr>().getInt() <= 0) {
            return printTo(emitError(), "Got unsupported dimension value '{0}' in 'shape' for 'SubViewAttr'", dimAttr);
        }
    }

    return mlir::success();
}

//
// SubViewAttr::print
//

void vpux::Const::SubViewAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getOffset());
    printer << ", ";
    printer.printAttribute(getShape());
    printer << ">";
}

//
// SubViewAttr::parse
//

mlir::Attribute vpux::Const::SubViewAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr offset;
    if (mlir::failed(parser.parseAttribute(offset))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr shape;
    if (mlir::failed(parser.parseAttribute(shape))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::SubViewAttr>(offset, shape);
}

//
// SubViewAttr::inferOutputType
//

mlir::ShapedType vpux::Const::SubViewAttr::inferOutputType(mlir::ShapedType input) const {
    const auto shape = parseIntArrayAttr(getShape());

    VPUX_THROW_UNLESS(shape.size() == checked_cast<size_t>(input.getRank()),
                      "View shape and input shape are not consistent in 'SubViewAttr'");

    return changeShape(input, ShapeRef(shape));
}

//
// SubViewAttr::transform
//

Const::Content vpux::Const::SubViewAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(),
                                                  input.isSplat());

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();

    if (input.isSplat()) {
        std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());
    } else {
        const Byte elemSize = getElemTypeSize(input.getStorageElemType());
        const auto order = DimsOrder::fromType(input.getType());

        const auto inShape = vpux::getShape(input.getType());
        const auto inMemShape = order.toMemoryOrder(inShape);

        const auto outShape = vpux::getShape(output.getType());
        const auto outMemShape = order.toMemoryOrder(outShape);

        const auto offset = Shape(parseIntArrayAttr(getOffset()));
        const auto memOffset = order.toMemoryOrder(offset);

        if (memOffset.size() == 1) {
            // Opitimized 1D case

            std::copy_n(inBuf.data() + memOffset.front() * elemSize.count(),
                        checked_cast<size_t>(output.getNumElements() * elemSize.count()), outBuf.data());
        } else if (memOffset.size() == 2) {
            // Opitimized 2D case

            const auto md0 = MemDim(0);
            const auto md1 = MemDim(1);

            const auto OUT0 = outMemShape[md0];
            const auto OUT1 = outMemShape[md1];

            const auto IN1 = inMemShape[md1];

            const auto off0 = memOffset[md0];
            const auto off1 = memOffset[md1];

            loop_2d(LoopExecPolicy::Parallel, OUT0, OUT1, [&](int64_t out0, int64_t out1) {
                const auto in0 = out0 + off0;
                const auto in1 = out1 + off1;

                const auto outRawInd = out1 + out0 * OUT1;
                const auto inRawInd = in1 + in0 * IN1;

                std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                            checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
            });
        } else if (memOffset.size() == 3) {
            // Opitimized 3D case

            const auto md0 = MemDim(0);
            const auto md1 = MemDim(1);
            const auto md2 = MemDim(2);

            const auto OUT0 = outMemShape[md0];
            const auto OUT1 = outMemShape[md1];
            const auto OUT2 = outMemShape[md2];

            const auto IN1 = inMemShape[md1];
            const auto IN2 = inMemShape[md2];

            const auto off0 = memOffset[md0];
            const auto off1 = memOffset[md1];
            const auto off2 = memOffset[md2];

            loop_3d(LoopExecPolicy::Parallel, OUT0, OUT1, OUT2, [&](int64_t out0, int64_t out1, int64_t out2) {
                const auto in0 = out0 + off0;
                const auto in1 = out1 + off1;
                const auto in2 = out2 + off2;

                const auto outRawInd = out2 + out1 * OUT2 + out0 * OUT2 * OUT1;
                const auto inRawInd = in2 + in1 * IN2 + in0 * IN2 * IN1;

                std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                            checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
            });
        } else if (memOffset.size() == 4) {
            // Opitimized 4D case

            const auto md0 = MemDim(0);
            const auto md1 = MemDim(1);
            const auto md2 = MemDim(2);
            const auto md3 = MemDim(3);

            const auto OUT0 = outMemShape[md0];
            const auto OUT1 = outMemShape[md1];
            const auto OUT2 = outMemShape[md2];
            const auto OUT3 = outMemShape[md3];

            const auto IN1 = inMemShape[md1];
            const auto IN2 = inMemShape[md2];
            const auto IN3 = inMemShape[md3];

            const auto off0 = memOffset[md0];
            const auto off1 = memOffset[md1];
            const auto off2 = memOffset[md2];
            const auto off3 = memOffset[md3];

            loop_4d(LoopExecPolicy::Parallel, OUT0, OUT1, OUT2, OUT3,
                    [&](int64_t out0, int64_t out1, int64_t out2, int64_t out3) {
                        const auto in0 = out0 + off0;
                        const auto in1 = out1 + off1;
                        const auto in2 = out2 + off2;
                        const auto in3 = out3 + off3;

                        const auto outRawInd = out3 + out2 * OUT3 + out1 * OUT3 * OUT2 + out0 * OUT3 * OUT2 * OUT1;
                        const auto inRawInd = in3 + in2 * IN3 + in1 * IN3 * IN2 + in0 * IN3 * IN2 * IN1;

                        std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                                    checked_cast<size_t>(elemSize.count()),
                                    outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
                    });
        } else {
            // Generic case

            loop_1d(LoopExecPolicy::Parallel, output.getNumElements(), [&](int64_t outMemInd1D) {
                const auto outMemIndND = getMemIndexND(outMemInd1D, outMemShape);

                MemShape inMemIndND(outMemIndND.size());
                for (auto ind : irange(inMemIndND.size())) {
                    const auto md = MemDim(ind);
                    inMemIndND[md] = outMemIndND[md] + memOffset[md];
                }

                const auto inMemInd1D = getMemIndex1D(inMemIndND, inMemShape);

                const auto inMemRawInd = checked_cast<size_t>(inMemInd1D * elemSize.count());
                VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'SubViewAttr'");

                const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
                VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'SubViewAttr'");

                std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + outMemRawInd);
            });
        }
    }

    return output;
}

//
// ContentAttr::subview
//

Const::ContentAttr vpux::Const::ContentAttr::subview(ShapeRef offset, ShapeRef shape) const {
    return get(*this,
               Const::SubViewAttr::get(getInt64ArrayAttr(getContext(), offset), getInt64ArrayAttr(getContext(), shape))
                       .cast<Const::TransformAttrInterface>());
}
