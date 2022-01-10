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

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <bits/stdint-intn.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

//
// ClampAttr::walkImmediateSubElements
//

void vpux::Const::ClampAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                      llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getAxis());
    walkAttrsFn(getMin());
    walkAttrsFn(getMax());
}

//
// ClampAttr::print
//

void vpux::Const::ClampAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getAxis());
    printer << ", ";
    printer.printAttribute(getMin());
    printer << ", ";
    printer.printAttribute(getMax());
    printer << ">";
}

//
// ClampAttr::verify
//

mlir::LogicalResult vpux::Const::ClampAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                   mlir::IntegerAttr axis, mlir::ArrayAttr min, mlir::ArrayAttr max) {
    if (min == nullptr) {
        return printTo(emitError(), "Got NULL 'min' in 'ClampAttr'");
    }
    if (max == nullptr) {
        return printTo(emitError(), "Got NULL 'max' in 'ClampAttr'");
    }

    if (min.size() != max.size()) {
        return printTo(emitError(), "Got inconsistent 'min' and 'max' values in 'ClampAttr'");
    }

    return mlir::success();
}

//
// ClampAttr::parse
//

mlir::Attribute vpux::Const::ClampAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr axis;
    if (mlir::failed(parser.parseAttribute(axis))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr min;
    if (mlir::failed(parser.parseAttribute(min))) {
        return nullptr;
    }

    mlir::ArrayAttr max;
    if (mlir::failed(parser.parseAttribute(max))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::ClampAttr::get(axis, min, max);
}

//
// ClampAttr::inferOutputType
//

mlir::ShapedType vpux::Const::ClampAttr::inferOutputType(mlir::ShapedType input) const {
    return input;
}

//
// ClampAttr::transform
//

Const::Content vpux::Const::ClampAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                  mlir::Float32Type::get(getContext()), input.isSplat());

    const auto values = input.getValues<float>();
    const auto shape = input.getShape();
    auto clampedVals = output.getTempBuf<float>();

    const auto axis = checked_cast<int64_t>(getAxis().getValue().getSExtValue());
    const auto min_value = parseFPArrayAttr<float>(getMin());
    const auto max_value = parseFPArrayAttr<float>(getMax());

    const auto N = shape[Dims4D::Act::N];
    const auto C = shape[Dims4D::Act::C];
    const auto H = shape[Dims4D::Act::H];
    const auto W = shape[Dims4D::Act::W];

    loop_1d(LoopExecPolicy::Sequential, clampedVals.size(), [&](size_t i) {
        auto ind = i;
        auto n = ind / (C * H * W);
        ind %= (C * H * W);
        auto c = ind / (H * W);
        ind %= (H * W);
        auto h = ind / W;
        ind %= W;
        auto w = ind;

        size_t quantInd;

        switch (axis) {
        case 0:
            quantInd = n;
            break;
        case 1:
            quantInd = c;
            break;
        case 2:
            quantInd = h;
            break;
        case 3:
            quantInd = w;
            break;
        default:
            VPUX_THROW("Axis has unexpected value {0}", axis);
        }

        clampedVals[i] = std::min<float>(std::max<float>(values[i], min_value[quantInd]), max_value[quantInd]);
        if (clampedVals[i] != values[i])
            std::cout << "ONO POMENYALOSSSSSSS\n";
    });

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::clamp(int32_t axis, vpux::Const::details::ContentRange<float> min,
                                                   vpux::Const::details::ContentRange<float> max) const {
    return get(*this, Const::ClampAttr::get(getIntAttr(getContext(), axis), getFPArrayAttr(getContext(), min),
                                            getFPArrayAttr(getContext(), max))
                              .cast<Const::TransformAttrInterface>());
}
