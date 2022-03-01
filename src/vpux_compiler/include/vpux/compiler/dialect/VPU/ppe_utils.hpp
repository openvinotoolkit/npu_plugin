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

#include <llvm/ADT/Optional.h>
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/pwl_utils.hpp"
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

llvm::Optional<double> calculateQuantScaleVectorForEltwise(mlir::ShapedType input1ShapedType,
                                                           mlir::ShapedType input2ShapedType,
                                                           mlir::ShapedType outputShapedType, VPU::ArchKind arch,
                                                           bool isMultiplyOp);
llvm::Optional<double> calculateQuantScaleVectorForAvgPool(mlir::ShapedType inputShapedType,
                                                           mlir::ShapedType outputShapedType,
                                                           ArrayRef<int64_t> filter_size, VPU::ArchKind arch);

VPU::PPETaskAttr getPPEAttr(VPU::PostOpParams postOpParams, mlir::MLIRContext* ctx);

VPU::PPETaskAttr getPPETaskAttrFromPostOpsParams(mlir::Value opInput, mlir::Value opOutput, vpux::IE::PostOp postOpAttr,
                                                 mlir::Location loc, mlir::MLIRContext* ctx, VPU::ArchKind arch);

VPU::PPETaskAttr getNCEEltwisePPETaskAttr(mlir::Value opInput1, mlir::Value opInput2, mlir::Value opOutput,
                                          vpux::IE::PostOp postOpAttr, mlir::Location loc, VPU::EltwiseType opType,
                                          mlir::MLIRContext* ctx, VPU::ArchKind arch);

PostOpParams getPwlPostOpParams(const mlir::Type inElemType, const mlir::Type outElemType, VPU::PPEMode ppeType);

llvm::Optional<PostOpParams> parsePostOp(IE::PostOp postOp, const mlir::Type inElemType, const mlir::Type outElemType,
                                         VPU::ArchKind arch, mlir::Location loc);

}  // namespace VPU
}  // namespace vpux
