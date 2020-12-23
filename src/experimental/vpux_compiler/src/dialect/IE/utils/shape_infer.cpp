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

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

mlir::FailureOr<SmallVector<int64_t, 4>> vpux::IE::broadcastEltwiseShape(ArrayRef<int64_t> shape1,
                                                                         ArrayRef<int64_t> shape2,
                                                                         AutoBroadcastType broadcastType,
                                                                         mlir::Location loc) {
    if (broadcastType == AutoBroadcastType::NONE_OR_EXPLICIT) {
        if (shape1 != shape2) {
            return mlir::LogicalResult(
                    printTo(mlir::emitError(loc), "Input shapes must be equal in case BroadcastType is NONE"));
        }

        return to_vector<4>(shape1);
    } else if (broadcastType == vpux::IE::AutoBroadcastType::NUMPY) {
        SmallVector<int64_t, 4> outShape(std::max(shape1.size(), shape2.size()), 0);

        auto in1ShapeIter = shape1.rbegin();
        auto in2ShapeIter = shape2.rbegin();

        for (auto outShapeRIter = outShape.rbegin(); outShapeRIter != outShape.rend(); ++outShapeRIter) {
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

    return mlir::LogicalResult(printTo(mlir::emitError(loc), "Unsupported BroadcastType '{0}'", broadcastType));
}

mlir::FailureOr<SmallVector<int64_t, 4>> vpux::IE::broadcastEltwiseShape(ArrayRef<ArrayRef<int64_t>> shapes,
                                                                         AutoBroadcastType broadcastType,
                                                                         mlir::Location loc) {
    if (shapes.size() < 2) {
        return mlir::LogicalResult(
                printTo(mlir::emitError(loc), "Number of input shapes must be equal or greater than 2"));
    }

    if (broadcastType == vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        for (size_t i = 1; i < shapes.size(); ++i) {
            if (shapes[0] != shapes[i]) {
                return mlir::LogicalResult(
                        printTo(mlir::emitError(loc), "Input shapes must be equal in case BroadcastType is NONE"));
            }
        }

        return to_vector<4>(shapes[0]);
    } else {
        size_t rank = shapes[0].size();
        for (size_t i = 1; i < shapes.size(); ++i) {
            rank = std::max(rank, shapes[i].size());
        }

        SmallVector<int64_t, 4> outShape(rank, 0);
        for (size_t i = 0; i < outShape.size(); ++i) {
            *(outShape.rbegin() + i) = *(shapes[0].rbegin() + i);
        }

        for (size_t i = 1; i < shapes.size(); ++i) {
            auto in1ShapeIter = outShape.rbegin();
            auto in2ShapeIter = shapes[i].rbegin();

            for (auto outShapeRIter = outShape.rbegin(); outShapeRIter != outShape.rend(); ++outShapeRIter) {
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

    return mlir::LogicalResult(printTo(mlir::emitError(loc), "Unsupported BroadcastType '{0}'", broadcastType));
}
