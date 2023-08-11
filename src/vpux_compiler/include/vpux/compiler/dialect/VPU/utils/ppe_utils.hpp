//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/type_interfaces.hpp"

#include <llvm/ADT/Optional.h>
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/pwl_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace VPU {

struct QuantizationParams {
    SmallVector<int64_t> quantMult;
    SmallVector<int64_t> quantShift;
    int64_t postShift;
};

struct PostOpParams {
    VPU::PPEMode layerType;
    int64_t clampLow;
    int64_t clampHigh;
    int64_t LreluMult;
    int64_t LreluShift;
    Optional<QuantizationParams> quantParams;

    PostOpParams(VPU::PPEMode layerType, int64_t clampLow, int64_t clampHigh, int64_t LreluMult, int64_t LreluShift)
            : layerType(layerType),
              clampLow(clampLow),
              clampHigh(clampHigh),
              LreluMult(LreluMult),
              LreluShift(LreluShift) {
    }

    PostOpParams(VPU::PPEMode layerType, int64_t clampLow, int64_t clampHigh, int64_t LreluMult, int64_t LreluShift,
                 const QuantizationParams& quantParams)
            : layerType(layerType),
              clampLow(clampLow),
              clampHigh(clampHigh),
              LreluMult(LreluMult),
              LreluShift(LreluShift),
              quantParams(quantParams) {
    }
};

double calculateQuantScaleVectorForEltwise(vpux::NDTypeInterface input1ShapedType,
                                           vpux::NDTypeInterface input2ShapedType,
                                           vpux::NDTypeInterface outputShapedType, VPU::ArchKind arch,
                                           bool isMultiplyOp);
double calculateQuantScaleVectorForAvgPool(vpux::NDTypeInterface inputShapedType,
                                           vpux::NDTypeInterface outputShapedType, ArrayRef<int64_t> filter_size,
                                           VPU::ArchKind arch);

VPU::PPETaskAttr getPPEAttr(VPU::PostOpParams postOpParams, mlir::MLIRContext* ctx);

VPU::PPETaskAttr getPPETaskAttrFromPostOpsParams(mlir::Value opInput, mlir::Value opOutput, vpux::IE::PostOp postOpAttr,
                                                 mlir::Location loc, mlir::MLIRContext* ctx, VPU::ArchKind arch);

VPU::PPETaskAttr getNCEAveragePoolPPETaskAttr(vpux::NDTypeInterface inputType, mlir::ArrayAttr kernelSizeAttr,
                                              vpux::NDTypeInterface outputType, vpux::IE::PostOp postOpAttr,
                                              mlir::Location loc, mlir::MLIRContext* ctx, VPU::ArchKind arch);

VPU::PPETaskAttr getNCEEltwisePPETaskAttr(vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                                          vpux::NDTypeInterface outputType, vpux::IE::PostOp postOpAttr,
                                          mlir::Location loc, VPU::EltwiseType opType, mlir::MLIRContext* ctx,
                                          VPU::ArchKind arch);

PostOpParams getPwlPostOpParams(const mlir::Type inElemType, const mlir::Type outElemType, VPU::PPEMode ppeType);

llvm::Optional<PostOpParams> parsePostOp(IE::PostOp postOp, const mlir::Type inElemType, const mlir::Type outElemType,
                                         VPU::ArchKind arch, mlir::Location loc);
bool supportsPerInputEltwiseScale(const VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
