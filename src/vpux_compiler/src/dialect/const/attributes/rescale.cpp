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
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// RescaleAttr::print
//

void vpux::Const::RescaleAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getScale());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::RescaleAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::FloatAttr scale;
    if (mlir::failed(parser.parseAttribute(scale))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::RescaleAttr::get(scale);
}

//
// RescaleAttr::inferOutputType
//

mlir::ShapedType vpux::Const::RescaleAttr::inferOutputType(mlir::ShapedType input) const {
    return input;
}

//
// RescaleAttr::transform
//

Const::Content vpux::Const::RescaleAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                  mlir::Float32Type::get(getContext()), input.isSplat());

    const auto values = input.getValues<float>();
    auto scaledVals = output.getTempBuf<float>();

    const auto scale = static_cast<float>(getScale().getValue().convertToDouble());

    loop_1d(LoopExecPolicy::Parallel, scaledVals.size(), [&](size_t i) {
        scaledVals[i] = values[i] * scale;
    });

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::rescale(double scale) const {
    return get(*this, Const::RescaleAttr::get(getFPAttr(getContext(), scale)).cast<Const::TransformAttrInterface>());
}
