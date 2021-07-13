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

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux {
namespace VPUIP {

class NCEInvariant final {
public:
    static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

public:
    static mlir::LogicalResult verifyOp(mlir::Operation* op, Logger log = Logger::global());

public:
    static mlir::LogicalResult verifyCMX(IERT::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyConvCMX(mlir::Location loc, mlir::ModuleOp module, mlir::MemRefType inputType,
                                             mlir::MemRefType filterType, mlir::MemRefType outputType,
                                             Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IERT::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module, mlir::MemRefType inputType,
                                             mlir::MemRefType outputType, mlir::ArrayAttr kernelSize,
                                             mlir::ArrayAttr kernelStrides, Logger log = Logger::global());

    static mlir::LogicalResult verifyCMX(IERT::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module,
                                                mlir::MemRefType firstInputType, mlir::MemRefType secondInputType,
                                                mlir::MemRefType outputType, Logger log = Logger::global());

public:
    static mlir::LogicalResult verifyChannels(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyConvChannels(mlir::Location loc, mlir::ShapedType filterType,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IERT::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwiseChannels(mlir::Location loc, mlir::ShapedType firstInputType,
                                                     mlir::ShapedType secondInputType, Logger log = Logger::global());

    static int64_t getChannelAlignment(mlir::Type elemType);

public:
    static mlir::LogicalResult verifyKernel(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::ConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::MaxPoolOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyKernel(IERT::AddOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyKernel(mlir::Location loc, mlir::ArrayAttr kernelSize,
                                            mlir::ArrayAttr kernelStrides, Logger log = Logger::global());
};

}  // namespace VPUIP
}  // namespace vpux
