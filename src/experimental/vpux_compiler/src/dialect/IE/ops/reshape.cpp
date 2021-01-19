//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <numeric>

using namespace vpux;

namespace {

mlir::FailureOr<SmallVector<int64_t>> getReshapeOutputShape(mlir::Location loc, IE::ReshapeOpAdaptor reshape) {
    auto shapeConst = reshape.shape().getDefiningOp<ConstantInterface>();
    if (shapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for shape");
    }

    auto shapeVec = to_small_vector(shapeConst.getContent().getValues<int64_t>());

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
        const auto inShape = to_small_vector(reshape.input().getType().cast<mlir::ShapedType>().getShape());

        auto dividend = std::accumulate(inShape.begin(), inShape.end(), int64_t(1), std::multiplies<int64_t>());

        for (size_t i = 0; i < shapeVec.size(); ++i) {
            auto& v = shapeVec[i];

            if (v == 0 && specialZero) {
                if (i < inShape.size()) {
                    v = inShape[i];
                } else {
                    return errorAt(loc, "Shape value at '{0}' is out of range '{1}'", i, inShape.size());
                }
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

mlir::LogicalResult vpux::IE::ReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReshapeOpAdaptor reshape(operands, attrs);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = getReshapeOutputShape(loc, reshape);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    inferredReturnShapes.emplace_back(outShape.getValue(),
                                      reshape.input().getType().cast<mlir::ShapedType>().getElementType());
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::ReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<ConstContentAttr>()) {
        return attr;
    }

    return nullptr;
}
