//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

bool isBroadcastable(int64_t d0, int64_t d1) {
    return d0 == 1 || d1 == 1 || d0 == d1;
}

}  // namespace

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::broadcastEltwiseShape(ArrayRef<int64_t> shape1,
                                                                      ArrayRef<int64_t> shape2,
                                                                      AutoBroadcastType broadcastType,
                                                                      mlir::Location loc) {
    if (broadcastType == IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        if (shape1 != shape2) {
            return errorAt(loc, "Input shapes must be equal in case BroadcastType is NONE");
        }

        return to_small_vector(shape1);
    } else if (broadcastType == IE::AutoBroadcastType::NUMPY) {
        SmallVector<int64_t> outShape(std::max(shape1.size(), shape2.size()), 0);

        auto in1ShapeIter = shape1.rbegin();
        auto in2ShapeIter = shape2.rbegin();

        for (auto outShapeRIter = outShape.rbegin(); outShapeRIter != outShape.rend(); ++outShapeRIter) {
            if (in1ShapeIter != shape1.rend() && in2ShapeIter != shape2.rend()) {
                if (!isBroadcastable(*in1ShapeIter, *in2ShapeIter)) {
                    return errorAt(loc, "Got non broadcastable dimensions pair : '{0}' and {1}'", *in1ShapeIter,
                                   *in2ShapeIter);
                }
            }

            *outShapeRIter = std::max(in1ShapeIter != shape1.rend() ? *in1ShapeIter : 0,
                                      in2ShapeIter != shape2.rend() ? *in2ShapeIter : 0);

            if (in1ShapeIter != shape1.rend()) {
                ++in1ShapeIter;
            }
            if (in2ShapeIter != shape2.rend()) {
                ++in2ShapeIter;
            }
        }

        return outShape;
    }

    return errorAt(loc, "Unsupported BroadcastType '{0}'", broadcastType);
}

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::broadcastEltwiseShape(ArrayRef<ArrayRef<int64_t>> shapes,
                                                                      AutoBroadcastType broadcastType,
                                                                      mlir::Location loc) {
    if (shapes.size() < 2) {
        return errorAt(loc, "Number of input shapes must be equal or greater than 2");
    }

    if (broadcastType == vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        for (size_t i = 1; i < shapes.size(); ++i) {
            if (shapes[0] != shapes[i]) {
                return errorAt(loc, "Input shapes must be equal in case BroadcastType is NONE");
            }
        }

        return to_small_vector(shapes[0]);
    } else {
        size_t rank = shapes[0].size();
        for (size_t i = 1; i < shapes.size(); ++i) {
            rank = std::max(rank, shapes[i].size());
        }

        SmallVector<int64_t> outShape(rank, 0);
        for (size_t i = 0; i < outShape.size(); ++i) {
            *(outShape.rbegin() + i) = *(shapes[0].rbegin() + i);
        }

        for (size_t i = 1; i < shapes.size(); ++i) {
            auto in1ShapeIter = outShape.rbegin();
            auto in2ShapeIter = shapes[i].rbegin();

            for (auto outShapeRIter = outShape.rbegin(); outShapeRIter != outShape.rend(); ++outShapeRIter) {
                if (in1ShapeIter != outShape.rend() && in2ShapeIter != shapes[i].rend()) {
                    if (!isBroadcastable(*in1ShapeIter, *in2ShapeIter)) {
                        return errorAt(loc, "Got non broadcastable dimensions pair : '{0}' and {1}'", *in1ShapeIter,
                                       *in2ShapeIter);
                    }
                }

                *outShapeRIter = std::max(in1ShapeIter != outShape.rend() ? *in1ShapeIter : 0,
                                          in2ShapeIter != shapes[i].rend() ? *in2ShapeIter : 0);

                if (in1ShapeIter != outShape.rend()) {
                    ++in1ShapeIter;
                }
                if (in2ShapeIter != shapes[i].rend()) {
                    ++in2ShapeIter;
                }
            }
        }

        return outShape;
    }

    return errorAt(loc, "Unsupported BroadcastType '{0}'", broadcastType);
}

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::constInputToData(mlir::Location loc, const mlir::Value& value) {
    if (value == nullptr) {
        return errorAt(loc, "Target shape was not provided");
    }

    auto valueConst = value.getDefiningOp<Const::DeclareOp>();
    if (valueConst == nullptr) {
        return errorAt(loc, "Only constant input is supported");
    }

    const auto valueContent = valueConst.content();
    return to_small_vector(valueContent.getValues<int64_t>());
}
